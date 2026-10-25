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

"""The bucket-level XML multi-object delete protocol."""

import base64
import binascii
import hashlib
import re
import xml.etree.ElementTree as ET
import zlib
from xml.parsers import expat

import flask

from testbench.error import RestException

NAMESPACE = "http://s3.amazonaws.com/doc/2006-03-01/"


def _response(element, status=200):
    # ElementTree emits literal CRs in text, which XML parsers normalize to LF.
    # Character references preserve the object key in the response as well.
    data = ET.tostring(element, encoding="utf-8", xml_declaration=True).replace(
        b"\r", b"&#13;"
    )
    return flask.Response(data, status=status, content_type="application/xml")


class XmlError(RestException):
    def __init__(self, code, message, status=400):
        super().__init__(message, status)
        self.xml_code = code

    def as_response(self):
        error = ET.Element("Error")
        ET.SubElement(error, "Code").text = self.xml_code
        ET.SubElement(error, "Message").text = self.msg
        return _response(error, self.code)

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


def _malformed():
    return XmlError(
        "MalformedMultiObjectDeleteRequest",
        "The XML is not a valid multi-object delete request.",
    )


def _validate_checksum(data, headers):
    md5_values = []
    if "Content-MD5" in headers:
        md5_values.append(headers["Content-MD5"])
    for value in headers.get("x-goog-hash", "").split(","):
        value = value.strip()
        if value.startswith("md5="):
            md5_values.append(value[4:])
    crc = headers.get("x-amz-checksum-crc32")
    if not md5_values and crc is None:
        raise XmlError(
            "InvalidRequest", "An MD5 or CRC32 request checksum is required."
        )
    checks = [(v, hashlib.md5(data).digest(), "InvalidDigest") for v in md5_values]
    if crc is not None:
        checks.append((crc, zlib.crc32(data).to_bytes(4, "big"), "InvalidRequest"))
    for value, expected, invalid_code in checks:
        try:
            digest = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error):
            raise XmlError(invalid_code, "The request checksum is invalid.")
        if len(digest) != len(expected):
            raise XmlError(invalid_code, "The request checksum is invalid.")
        if digest != expected:
            raise XmlError("BadDigest", "The request checksum does not match the body.")


def _parse_xml(data):
    # Retain namespace prefixes: GCS validates the namespace but rejects
    # prefixed request elements, even when the namespace URI is correct.
    parser = expat.ParserCreate(namespace_separator=" ")
    parser.namespace_prefixes = True
    tree = ET.TreeBuilder()
    prefixed = set()

    def expanded(name):
        parts = name.split(" ")
        return "{%s}%s" % tuple(parts[:2]) if len(parts) > 1 else name

    def start(name, attributes):
        if len(name.split(" ")) == 3:
            prefixed.add(expanded(name))
        tree.start(expanded(name), attributes)

    def doctype(*args):
        # Reject before any entity expansion, including in UTF-16 documents.
        raise _malformed()

    parser.StartElementHandler = start
    parser.EndElementHandler = lambda name: tree.end(expanded(name))
    parser.CharacterDataHandler = tree.data
    parser.StartDoctypeDeclHandler = doctype
    try:
        parser.Parse(data, True)
    except expat.ExpatError:
        raise _malformed()
    return tree.close(), prefixed


def _parse_request(request):
    data = request.get_data()
    _validate_checksum(data, request.headers)
    root, prefixed = _parse_xml(data)
    namespace = "{%s}" % NAMESPACE if root.tag.startswith("{") else ""
    if root.tag != namespace + "Delete":
        raise _malformed()
    for element in root.iter():
        if element.attrib:
            raise _malformed()
        if element.tag in (namespace + "Delete", namespace + "Object"):
            if (element.text or "").strip() or any(
                (c.tail or "").strip() for c in element
            ):
                raise _malformed()
    entries = []
    quiet = False
    seen_quiet = False
    for child in root:
        if child.tag == namespace + "Quiet" and not seen_quiet:
            if len(child):
                raise _malformed()
            quiet = (child.text or "").lower() == "true"
            seen_quiet = True
            continue
        if child.tag != namespace + "Object":
            raise _malformed()
        fields = {}
        for field in child:
            if (
                field.tag not in (namespace + "Key", namespace + "VersionId")
                or field.tag in fields
                or len(field)
            ):
                raise _malformed()
            fields[field.tag] = field.text or ""
        key = fields.get(namespace + "Key")
        version = fields.get(namespace + "VersionId")
        if key is None:
            raise _malformed()
        if version is not None:
            if not re.fullmatch(r"[+-]?[0-9]+", version.strip()):
                raise _malformed()
            negative = version.strip().startswith("-")
            version = version.strip().lstrip("+-").lstrip("0") or "0"
            if negative and version != "0":
                raise _malformed()
            if len(version) > 19 or (
                len(version) == 19 and version > "9223372036854775807"
            ):
                raise _malformed()
        # The live XML endpoint trims surrounding XML whitespace from keys.
        # Preserve all other text, including percent escapes and internal spaces.
        entries.append((key.strip(" \t\r\n"), version))
    if not entries:
        raise _malformed()
    if len(entries) > 1000:
        raise XmlError(
            "InvalidMultiObjectDeleteRequest", "At most 1000 objects can be deleted."
        )
    if prefixed:
        if namespace + "Key" in prefixed and namespace + "Object" not in prefixed:
            raise XmlError("InvalidObjectName", "An object name is required.")
        raise XmlError("InvalidArgument", "Prefixed XML elements are not supported.")
    return entries, quiet


def delete_objects(database, request, bucket_name):
    entries, quiet = _parse_request(request)
    # A missing bucket is a request failure, even when every key is absent.
    database.get_bucket(bucket_name, None)
    if any(not key for key, _ in entries):
        raise XmlError("InvalidObjectName", "An object name is required.")
    result = ET.Element("DeleteResult", xmlns=NAMESPACE)
    for key, version in dict.fromkeys(entries):
        error = None
        response_version = version
        try:
            generation = int(version) if version is not None else 0
            if version == "0":
                # Explicit zero selects the current generation for removal, even
                # in a versioned bucket. Resolve it to an explicit generation so
                # a concurrent replacement cannot be deleted by this request.
                generation = database.get_object(bucket_name, key).metadata.generation
            database.delete_object(
                bucket_name,
                key,
                generation=generation,
                preconditions=[_check_hold],
            )
            if version == "0":
                response_version = str(generation)
        except RestException as ex:
            if int(ex.code) == 404:
                # Confirm this was a missing object rather than a bucket removed
                # concurrently with the request. Missing objects are successful.
                database.get_bucket(bucket_name, None)
            else:
                error = XmlError.from_rest(ex)
        if error is None and quiet:
            continue
        item = ET.SubElement(result, "Error" if error else "Deleted")
        ET.SubElement(item, "Key").text = key
        if response_version is not None:
            ET.SubElement(item, "Generation").text = response_version
            ET.SubElement(item, "VersionId").text = response_version
        if error:
            ET.SubElement(item, "Code").text = error.xml_code
            ET.SubElement(item, "Message").text = error.msg
    return _response(result)


def _check_hold(blob, live_generation, context):
    # Database preconditions run under its resource lock, so a held object cannot
    # change between checking its metadata and deleting the selected generation.
    if blob.metadata.temporary_hold or blob.metadata.event_based_hold:
        raise XmlError(
            "ObjectUnderActiveHold", "The object is under an active hold.", 403
        )
    return True
