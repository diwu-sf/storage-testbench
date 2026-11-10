# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The bucket-level XML multi-object delete protocol.

Handles `POST /BUCKET?delete` (and the bucket subdomain's `POST /?delete`), see
https://cloud.google.com/storage/docs/xml-api/post-bucket. The behavior,
including error codes, messages, and precedence, mirrors what live GCS returned
when probed on 2026-11-10:

- Request errors are reported in this order: checksum headers, XML syntax and
  schema (`MalformedMultiObjectDeleteRequest`), mismatched `VersionId` and
  `Generation`, no objects, more than 1,000 objects (counted before duplicates
  are removed), a missing bucket (`NoSuchBucket`), and finally an empty key
  (`InvalidObjectName`). A rejected request deletes nothing.
- The request needs a Base64 MD5 of the body in `Content-MD5` or
  `x-goog-hash: md5=...`, or a Base64 big-endian CRC32 in
  `x-amz-checksum-crc32`. CRC32C alone is not enough. All supplied MD5 and
  CRC32 checksums are checked.
- The body is always decoded as UTF-8: encoding declarations are ignored and
  invalid bytes become U+FFFD. Byte order marks, and therefore UTF-16 bodies,
  are rejected. Internal DTD subsets and entities are accepted. External DTDs
  and entities are never loaded: documents that declare external or parameter
  entities, or reference entities an external DTD would define, are rejected.
- The document may use no namespace, or the default namespace `NAMESPACE`.
  Namespace prefixes pass validation but GCS ignores prefixed child elements:
  a prefixed `VersionId`, `Generation` or `Quiet` is ignored, a prefixed `Key`
  leaves the object unnamed, and a prefixed `Object` is skipped. If an
  unprefixed field is inside a prefixed `Object`, GCS fails with an empty 503.
- Keys are trimmed of whitespace as defined by Guava's
  `CharMatcher.whitespace()`. Carriage returns in keys are returned unescaped.
- `Quiet` suppresses `Deleted` entries only when its text equals `true`
  ignoring case. Missing objects and generations are reported as `Deleted`.
  The response lists `Deleted` entries and then `Error` entries, each sorted by
  key.
- Identical targets are collapsed, with generations compared numerically.
  Explicit generations are echoed in both `Generation` and `VersionId`.
  Explicit generation zero removes the live generation (even in a versioned
  bucket) and reports that generation, or `0` if there is none. A key named by
  several entries also reports the generation each `Deleted` entry resolved
  to. If two entries resolve to the same existing generation and both
  deletions succeed, GCS still applies every deletion but reports
  `InternalError` for every entry that targeted an existing object, including
  unrelated keys. The final state then depends on a race in GCS; the testbench
  applies entries in request order.
- Holds and unexpired retention produce per-object `ObjectUnderActiveHold` and
  `RetentionPolicyNotMet` errors.

Known differences: schema error diagnostics are not reproduced, IAM is not
enforced, and XML 1.1 documents are not supported.
"""

import base64
import binascii
import collections
import datetime
import hashlib
import re
import zlib
from xml.parsers import expat

import flask

from testbench.error import RestException

NAMESPACE = "http://s3.amazonaws.com/doc/2006-03-01/"
MAX_OBJECTS = 1000
MAX_GENERATION = 2**63 - 1

# GCS trims object names with Guava's `CharMatcher.whitespace()`. Notably this
# includes NBSP and other Unicode spaces, but not U+200B, U+180E, or U+FEFF.
_KEY_WHITESPACE = (
    "\t\n\u000b\u000c\r \u0085\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)
_XML_WHITESPACE = " \t\r\n"

_MALFORMED = (
    "The XML you provided was not well-formed or did not validate against our "
    "published schema."
)
_MISSING_CHECKSUM = (
    "Missing required header for this request: Content-MD5 OR x-amz-checksum-crc32."
)
_MD5_MISMATCH = (
    "The MD5 you specified in Content-MD5 or x-goog-hash did not match what we "
    "computed."
)


def _escape(text):
    # Like GCS, emit carriage returns literally rather than as `&#13;`.
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _response(body, status=200):
    data = ("<?xml version='1.0' encoding='UTF-8'?>" + body).encode("utf-8")
    return flask.Response(
        data, status=status, content_type="application/xml; charset=UTF-8"
    )


def _element(tag, text):
    return "<%s>%s</%s>" % (tag, _escape(text), tag)


class XmlError(RestException):
    def __init__(self, code, message, status=400, details=None):
        super().__init__(message, status)
        self.xml_code = code
        self.details = details

    def as_response(self):
        if self.xml_code is None:
            return flask.Response(b"", status=self.code)
        body = _element("Code", self.xml_code) + _element("Message", self.msg)
        if self.details is not None:
            body += _element("Details", self.details)
        return _response("<Error>%s</Error>" % body, self.code)

    @classmethod
    def from_rest(cls, error):
        if isinstance(error, cls):
            return error
        status = int(error.code)
        code = {
            400: "InvalidArgument",
            401: "AuthenticationRequired",
            403: "AccessDenied",
            404: "NoSuchBucket",
            408: "RequestTimeout",
            409: "OperationAborted",
            412: "PreconditionFailed",
            429: "SlowDown",
            500: "InternalError",
            501: "NotImplemented",
            503: "SlowDown",
        }.get(status, "InternalError")
        return cls(code, str(error.msg), status)


def _malformed(details=None):
    return XmlError("MalformedMultiObjectDeleteRequest", _MALFORMED, details=details)


def _invalid_request(details):
    return XmlError(
        "InvalidMultiObjectDeleteRequest",
        "The multi-object delete request was invalid.",
        details=details,
    )


def _no_such_bucket():
    return XmlError("NoSuchBucket", "The specified bucket does not exist.", 404)


# === CHECKSUMS === #


def _b64decode(value, size):
    # GCS accepts unpadded Base64 in checksum headers, but not URL-safe Base64.
    value += "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        return None
    return decoded if len(decoded) == size else None


def _validate_checksum(data, headers):
    goog_md5 = []
    for entry in headers.get("x-goog-hash", "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        kind, sep, value = entry.partition("=")
        if not sep:
            raise XmlError(
                "InvalidHash",
                "The x-goog-hash header you specified was invalid.",
                details="Invalid hash entry: " + entry,
            )
        if kind not in ("md5", "crc32c"):
            raise XmlError(
                "UnsupportedHash",
                "The x-goog-hash header you specified was invalid.",
                details="Unsupported hash type: " + kind,
            )
        # x-goog-hash CRC32C values are accepted but not checked by GCS.
        if kind == "md5":
            goog_md5.append(value)
    content_md5 = headers.get("Content-MD5")
    crc = headers.get("x-amz-checksum-crc32")
    if content_md5 is None and not goog_md5 and crc is None:
        raise XmlError("InvalidRequest", _MISSING_CHECKSUM, details=_MISSING_CHECKSUM)

    md5_values = ([content_md5] if content_md5 is not None else []) + goog_md5
    decoded = {}
    for value in md5_values:
        digest = _b64decode(value, 16)
        if digest is None:
            raise XmlError(
                "InvalidDigest",
                "The MD5 you specified in Content-MD5 or x-goog-hash was invalid.",
                details="Invalid MD5 value: " + value,
            )
        decoded[value] = digest
    if content_md5 is not None:
        for value in goog_md5:
            if decoded[value] != decoded[content_md5]:
                raise XmlError(
                    "BadDigest",
                    _MD5_MISMATCH,
                    details="The MD5 hashes in Content-MD5 (%s) and x-goog-hash (%s) don't match."
                    % (decoded[content_md5].hex(), decoded[value].hex()),
                )
    expected = hashlib.md5(data).digest()
    if any(digest != expected for digest in decoded.values()):
        raise XmlError(
            "BadDigest",
            _MD5_MISMATCH,
            details="The specified Content-MD5 does not match what we computed.",
        )
    if crc is not None:
        digest = _b64decode(crc, 4)
        if digest is None:
            raise XmlError(
                "InvalidRequest", "Value for x-amz-checksum-crc32 header is invalid."
            )
        if digest != zlib.crc32(data).to_bytes(4, "big"):
            raise XmlError(
                "BadDigest",
                "The CRC32 you specified did not match the calculated checksum.",
            )


# === PARSING === #


class _Node:
    __slots__ = ("namespace", "name", "prefixed", "children", "text")

    def __init__(self, namespace, name, prefixed):
        self.namespace = namespace
        self.name = name
        self.prefixed = prefixed
        self.children = []
        self.text = []


def _parse_xml(data):
    # GCS always decodes the body as UTF-8, replacing invalid sequences and
    # ignoring any encoding declaration. A byte order mark is not accepted.
    # Pyexpat silently drops NUL characters, which are never valid XML.
    text = data.decode("utf-8", errors="replace")
    if text.startswith("\ufeff") or "\x00" in text:
        raise _malformed()
    # Retain prefixes: GCS validates namespaces but only binds unprefixed
    # elements to the request.
    parser = expat.ParserCreate(namespace_separator=" ")
    parser.namespace_prefixes = True
    stack = []
    roots = []

    def start(name, attributes):
        if attributes:
            raise _malformed()
        parts = name.split(" ")
        if len(parts) == 1:
            node = _Node(None, parts[0], False)
        else:
            node = _Node(parts[0], parts[1], len(parts) == 3)
        (stack[-1].children if stack else roots).append(node)
        stack.append(node)

    def end(name):
        stack.pop()

    def characters(data):
        stack[-1].text.append(data)

    def reject(*args):
        raise _malformed()

    def entity(name, is_parameter, value, base, system_id, public_id, notation):
        # Expat (2.4.1+) limits internal entity amplification.
        if is_parameter or value is None or expat.version_info < (2, 4, 1):
            raise _malformed()

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = characters
    # DTDs are accepted, like GCS, but external resources are never loaded:
    # external entity declarations and references to entities that an
    # external subset might declare are rejected.
    parser.EntityDeclHandler = entity
    parser.UnparsedEntityDeclHandler = reject
    parser.ExternalEntityRefHandler = reject
    parser.SkippedEntityHandler = reject
    try:
        parser.Parse(text, True)
    except expat.ExpatError:
        raise _malformed()
    return roots[0]


def _parse_generation(text):
    value = text.strip(_XML_WHITESPACE)
    if not re.fullmatch(r"[+-]?[0-9]+", value):
        raise _malformed()
    digits = value.lstrip("+-").lstrip("0") or "0"
    if value.startswith("-") and digits != "0":
        raise _malformed()
    if len(digits) > 19 or int(digits) > MAX_GENERATION:
        raise _malformed()
    return int(digits)


def _validate_schema(root):
    """Check the document against the namespace-aware request schema.

    Prefixes are irrelevant here; every element must be in the same namespace
    as `Delete`, which is either no namespace or the S3 namespace.
    """
    if root.namespace not in (None, NAMESPACE) or root.name != "Delete":
        raise _malformed()

    def element_only(node, allowed):
        if "".join(node.text).strip(_XML_WHITESPACE):
            raise _malformed()
        counts = collections.Counter()
        for child in node.children:
            if child.namespace != root.namespace or child.name not in allowed:
                raise _malformed()
            counts[child.name] += 1
            if counts[child.name] > allowed[child.name]:
                raise _malformed()
        return counts

    def text_only(node):
        if node.children:
            raise _malformed()
        return "".join(node.text)

    counts = element_only(root, {"Quiet": 1, "Object": float("inf")})
    if counts["Object"] == 0:
        raise _malformed()
    for child in root.children:
        if child.name == "Quiet":
            text_only(child)
            continue
        if element_only(child, {"Key": 1, "Generation": 1, "VersionId": 1})["Key"] != 1:
            raise _malformed()
        for field in child.children:
            value = text_only(field)
            if field.name != "Key":
                _parse_generation(value)


class _Request:
    def __init__(self, entries, quiet):
        self.entries = entries
        self.quiet = quiet


def _bind(root):
    """Bind the validated document to a request, like the GCS XML binder.

    Only unprefixed elements are bound; prefixed ones are silently ignored.
    """
    quiet = False
    entries = []
    for child in root.children:
        if child.prefixed:
            # An unprefixed field inside an ignored Object makes GCS fail
            # with an empty 503 response.
            if child.name == "Object" and any(not f.prefixed for f in child.children):
                raise XmlError(None, "", 503)
            continue
        if child.name == "Quiet":
            quiet = "".join(child.text).lower() == "true"
            continue
        key, values = "", {}
        for field in child.children:
            if field.prefixed:
                continue
            if field.name == "Key":
                key = "".join(field.text)
            else:
                values[field.name] = _parse_generation("".join(field.text))
        if len(set(values.values())) > 1:
            raise _invalid_request(
                "Both VersionId and Generation are set for the same object and do not match."
            )
        generation = next(iter(values.values()), None)
        entries.append((key, generation))
    if not entries:
        raise XmlError(
            "InvalidArgument",
            "Invalid argument.",
            details="Specify at least one object name.",
        )
    if len(entries) > MAX_OBJECTS:
        raise _invalid_request("Too many objects; limit is %d." % MAX_OBJECTS)
    return _Request(entries, quiet)


def _parse_request(request):
    data = request.get_data()
    _validate_checksum(data, request.headers)
    root = _parse_xml(data)
    _validate_schema(root)
    return _bind(root)


# === DELETION === #


def _require_bucket(database, bucket_name):
    try:
        database.get_bucket(bucket_name, None)
    except RestException as ex:
        if int(ex.code) == 404:
            raise _no_such_bucket()
        raise


def _protection(bucket_name, key):
    def check(blob, live_generation, context):
        # Database preconditions run under its resource lock, so the object
        # cannot change between this check and the deletion.
        metadata = blob.metadata
        if metadata.temporary_hold or metadata.event_based_hold:
            kind = "Temporary" if metadata.temporary_hold else "Event-Based"
            raise XmlError(
                "ObjectUnderActiveHold",
                "Object '%s/%s' is under active %s hold and cannot be deleted, "
                "overwritten or archived until hold is removed."
                % (bucket_name, key, kind),
                403,
            )
        times = []
        if metadata.HasField("retention_expire_time"):
            times.append(metadata.retention_expire_time)
        if metadata.retention.HasField("retain_until_time"):
            times.append(metadata.retention.retain_until_time)
        utc = datetime.timezone.utc
        until = max((t.ToDatetime(tzinfo=utc) for t in times), default=None)
        if until is not None and until > datetime.datetime.now(utc):
            raise XmlError(
                "RetentionPolicyNotMet",
                "Object '%s/%s' is subject to bucket's retention policy or object "
                "retention and cannot be deleted or overwritten until %s"
                % (bucket_name, key, until.isoformat()),
                403,
            )
        return True

    return check


def _resolve(database, bucket_name, key, generation):
    """Return the generation an entry targets, or None if nothing exists."""
    try:
        if generation:
            database.get_object(bucket_name, key, generation=generation)
            return generation
        return database.get_object(bucket_name, key).metadata.generation
    except RestException as ex:
        if int(ex.code) != 404:
            raise
    # Confirm this was a missing object rather than a removed bucket.
    _require_bucket(database, bucket_name)
    return None


def _delete(database, bucket_name, key, generation, target):
    """Delete one entry. Returns an XmlError, or None on success."""
    if generation == 0:
        # Explicit zero removes the live generation, even in versioned
        # buckets. Use the resolved generation so a concurrent replacement is
        # never deleted by this request.
        if target is None:
            return None
        generation = target
    try:
        database.delete_object(
            bucket_name,
            key,
            generation=generation or 0,
            preconditions=[_protection(bucket_name, key)],
        )
    except RestException as ex:
        if int(ex.code) != 404:
            return XmlError.from_rest(ex)
        # Missing objects are successful, but a removed bucket is not.
        _require_bucket(database, bucket_name)
    return None


def delete_objects(database, request, bucket_name):
    parsed = _parse_request(request)
    _require_bucket(database, bucket_name)
    entries = [(key.strip(_KEY_WHITESPACE), gen) for key, gen in parsed.entries]
    if any(not key for key, _ in entries):
        raise XmlError("InvalidObjectName", "You must specify an object name.")
    # Identical targets (after normalization) are collapsed.
    entries = list(dict.fromkeys(entries))
    per_key = collections.Counter(key for key, _ in entries)
    targets = [_resolve(database, bucket_name, k, g) for k, g in entries]
    results = [
        _delete(database, bucket_name, key, generation, target)
        for (key, generation), target in zip(entries, targets)
    ]
    # When two entries resolve to the same existing object generation and both
    # deletions succeed, GCS keeps the deletions but reports InternalError for
    # every entry that targeted an existing object, including unrelated keys.
    successes = collections.Counter(
        (key, target)
        for (key, _), target, error in zip(entries, targets, results)
        if target is not None and error is None
    )
    conflict = any(count > 1 for count in successes.values())

    deleted, errors = [], []
    for (key, generation), target, error in zip(entries, targets, results):
        if conflict and target is not None:
            error = XmlError(
                "InternalError",
                "We encountered an internal error. Please try again.",
                500,
            )
        if error is not None or generation:
            # Errors and explicit generations echo the requested generation.
            reported = generation
        elif generation == 0 or per_key[key] > 1:
            # Explicit zero, or a key named by several entries, reports the
            # resolved live generation (zero when there is none).
            reported = target or 0
        else:
            reported = None
        item = _element("Key", key)
        if reported is not None:
            item += _element("Generation", str(reported))
            item += _element("VersionId", str(reported))
        if error is not None:
            item += _element("Code", error.xml_code) + _element("Message", error.msg)
            errors.append((key, "<Error>%s</Error>" % item))
        elif not parsed.quiet:
            deleted.append((key, "<Deleted>%s</Deleted>" % item))
    # GCS lists successes, then errors, each ordered by key.
    items = sorted(deleted, key=lambda i: i[0]) + sorted(errors, key=lambda i: i[0])
    body = "".join(item for _, item in items)
    root = "DeleteResult xmlns='%s'" % NAMESPACE
    return _response("<%s>%s</DeleteResult>" % (root, body) if body else "<%s/>" % root)
