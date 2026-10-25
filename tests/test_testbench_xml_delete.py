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

"""XML bulk deletion, including shared JSON/gRPC state and retry faults."""

import base64
import hashlib
import os
import unittest
import unittest.mock
import xml.etree.ElementTree as ET
import zlib
from urllib.parse import quote

import grpc

from google.storage.v2 import storage_pb2, storage_pb2_grpc
from testbench import grpc_server, rest_server
from testbench.error import RestException

NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


class TestXmlDelete(unittest.TestCase):
    def setUp(self):
        rest_server.db.clear()
        for app in (rest_server.root, rest_server.server):
            config = unittest.mock.patch.dict(
                app.config,
                SERVER_NAME="storage.googleapis.com",
                PREFERRED_URL_SCHEME="https",
            )
            config.start()
            self.addCleanup(config.stop)
        self.client = rest_server.server.test_client()
        env = unittest.mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("GOOGLE_CLOUD_CPP_STORAGE_TEST_BUCKET_NAME", None)
        response = self.client.post("/storage/v1/b", json={"name": "bucket-name"})
        self.assertEqual(response.status_code, 200, response.data)

    def put(self, name, data=b"contents"):
        response = self.client.post(
            "/upload/storage/v1/b/bucket-name/o",
            query_string={"uploadType": "media", "name": name},
            data=data,
        )
        self.assertEqual(response.status_code, 200, response.data)
        return response.json["generation"]

    def get(self, name, **query):
        return self.client.get(
            "/storage/v1/b/bucket-name/o/" + quote(name, safe=""), query_string=query
        )

    def body(self, entries, quiet=None):
        root = ET.Element("Delete")
        for key, version in entries:
            item = ET.SubElement(root, "Object")
            ET.SubElement(item, "Key").text = key
            if version is not None:
                ET.SubElement(item, "VersionId").text = str(version)
        if quiet is not None:
            ET.SubElement(root, "Quiet").text = quiet
        return ET.tostring(root, encoding="utf-8").replace(b"\r", b"&#13;")

    def post(self, body, path="/bucket-name?delete", headers=None, **kwargs):
        if isinstance(body, str):
            body = body.encode()
        if headers is None:
            headers = {
                "Content-MD5": base64.b64encode(hashlib.md5(body).digest()).decode()
            }
        return self.client.post(
            path, data=body, headers=headers, content_type="application/xml", **kwargs
        )

    def result(self, response):
        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(response.mimetype, "application/xml")
        root = ET.fromstring(response.data)
        self.assertEqual(root.tag, NS + "DeleteResult")
        return root

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.data)
        self.assertEqual(response.mimetype, "application/xml")
        error = ET.fromstring(response.data)
        self.assertEqual(error.tag, "Error")
        self.assertEqual(error.findtext("Code"), code)
        self.assertTrue(error.findtext("Message"))

    def test_checksums(self):
        body = self.body([("object", None)])
        md5 = base64.b64encode(hashlib.md5(body).digest()).decode()
        crc = base64.b64encode(zlib.crc32(body).to_bytes(4, "big")).decode()
        for headers in (
            {"Content-MD5": md5},
            {"x-goog-hash": "md5=" + md5},
            {"x-amz-checksum-crc32": crc},
            {"Content-MD5": md5, "x-amz-checksum-crc32": crc},
        ):
            with self.subTest(headers=headers):
                self.put("object")
                self.result(self.post(body, headers=headers))
                self.assertEqual(self.get("object").status_code, 404)
        self.put("object")
        for headers, code in (
            ({}, "InvalidRequest"),
            ({"x-goog-hash": "crc32c=AAAAAA=="}, "InvalidRequest"),
            ({"x-amz-checksum-crc32c": "AAAAAA=="}, "InvalidRequest"),
            ({"Content-MD5": "!"}, "InvalidDigest"),
            ({"Content-MD5": "YQ=="}, "InvalidDigest"),
            ({"Content-MD5": "AAAAAAAAAAAAAAAAAAAAAA=="}, "BadDigest"),
            ({"x-goog-hash": "md5=!"}, "InvalidDigest"),
            ({"x-amz-checksum-crc32": "!"}, "InvalidRequest"),
            ({"x-amz-checksum-crc32": "YQ=="}, "InvalidRequest"),
            ({"x-amz-checksum-crc32": "AAAAAA=="}, "BadDigest"),
            ({"Content-MD5": md5, "x-amz-checksum-crc32": "AAAAAA=="}, "BadDigest"),
            (
                {"Content-MD5": md5, "x-goog-hash": "md5=AAAAAAAAAAAAAAAAAAAAAA=="},
                "BadDigest",
            ),
        ):
            with self.subTest(headers=headers):
                self.assert_error(self.post(body, headers=headers), 400, code)
                self.assertEqual(self.get("object").status_code, 200)
        # Hash the exact received bytes, including whitespace after the document.
        self.assert_error(
            self.post(body + b" ", headers={"Content-MD5": md5}), 400, "BadDigest"
        )
        self.assertEqual(self.get("object").status_code, 200)

    def test_invalid_envelope_does_not_partially_delete(self):
        self.put("object")
        valid = "<Object><Key>object</Key></Object>"
        for body in (
            "",
            "<Delete/>",
            "<Delete>" + valid,
            "<Other>" + valid + "</Other>",
            "<Delete>" + valid + "<Object/></Delete>",
            "<Delete>" + valid + "<Other/></Delete>",
            "<Delete>" + valid + "<Quiet>True</Quiet><Quiet>False</Quiet></Delete>",
            "<Delete>" + valid + "<Object><Key>a</Key><Key>b</Key></Object></Delete>",
            "<Delete>"
            + valid
            + "<Object><Key>a</Key><VersionId>1</VersionId><VersionId>2</VersionId></Object></Delete>",
            "<Delete>" + valid + "<Object><Key><Nested/></Key></Object></Delete>",
            "<Delete>" + valid + "<Object><Key>a</Key><Other/></Object></Delete>",
            "<Delete attr='x'>" + valid + "</Delete>",
            "<Delete>text" + valid + "</Delete>",
            "<Delete>" + valid + "tail</Delete>",
            "<Delete><Object><Key attr='x'>object</Key></Object></Delete>",
            "<Delete>" + valid + "<Quiet><Nested/></Quiet></Delete>",
            "<Delete xmlns='urn:unknown'>" + valid + "</Delete>",
            "<!DOCTYPE Delete><Delete>" + valid + "</Delete>",
            "<!DOCTYPE Delete [<!ENTITY key 'object'>]><Delete><Object><Key>&key;</Key></Object></Delete>",
            "<!DOCTYPE Delete SYSTEM 'file:///not-read'><Delete>" + valid + "</Delete>",
        ):
            with self.subTest(body=body):
                self.assert_error(
                    self.post(body), 400, "MalformedMultiObjectDeleteRequest"
                )
                self.assertEqual(self.get("object").status_code, 200)
        body = (
            "<!DOCTYPE Delete [<!ENTITY key 'object'>]><Delete><Object><Key>&key;</Key></Object></Delete>"
        ).encode("utf-16")
        self.assert_error(self.post(body), 400, "MalformedMultiObjectDeleteRequest")
        self.assertEqual(self.get("object").status_code, 200)

    def test_oversized_batch(self):
        self.put("object")
        response = self.post(self.body([("object", None)] * 1001))
        self.assert_error(response, 400, "InvalidMultiObjectDeleteRequest")
        self.assertEqual(self.get("object").status_code, 200)

    def test_invalid_generations_reject_entire_request(self):
        self.put("object")
        for version in (
            "",
            "-1",
            "null",
            "abc",
            "1.5",
            "9223372036854775808",
            "١",
            "9" * 5000,
        ):
            with self.subTest(version=version):
                response = self.post(
                    self.body([("object", None), ("missing", version)])
                )
                self.assert_error(response, 400, "MalformedMultiObjectDeleteRequest")
                self.assertEqual(self.get("object").status_code, 200)

    def test_missing_bucket_is_not_missing_key(self):
        self.assert_error(
            self.post(self.body([("missing", None)]), "/missing-bucket?delete"),
            404,
            "NoSuchBucket",
        )

    def test_backend_programming_error_is_not_acknowledged(self):
        self.put("object")
        with unittest.mock.patch.object(
            rest_server.db, "delete_object", side_effect=RuntimeError("backend bug")
        ):
            response = self.post(self.body([("object", None)]))
        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.get("object").status_code, 200)

    def test_bucket_removed_during_request_is_not_acknowledged(self):
        delete = rest_server.db.delete_object

        def remove_bucket(bucket, key, **kwargs):
            rest_server.db.delete_bucket(bucket, None)
            return delete(bucket, key, **kwargs)

        with unittest.mock.patch.object(
            rest_server.db, "delete_object", side_effect=remove_bucket
        ):
            response = self.post(self.body([("missing", None)]))
        self.assert_error(response, 404, "NoSuchBucket")

    def test_namespace_and_quiet(self):
        for namespace in ("", ' xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'):
            for quiet in (
                "True",
                "true",
                "TRUE",
                "False",
                "false",
                "FALSE",
                "1",
                "0",
                "yes",
                "",
                " true ",
            ):
                with self.subTest(namespace=namespace, quiet=quiet):
                    self.put("object")
                    body = (
                        "<Delete%s><Quiet>%s</Quiet><Object><Key>object</Key></Object></Delete>"
                        % (namespace, quiet)
                    )
                    result = self.result(self.post(body))
                    self.assertEqual(len(result), 0 if quiet.lower() == "true" else 1)
                    self.assertEqual(self.get("object").status_code, 404)
        response = self.post(
            '<d:Delete xmlns:d="http://s3.amazonaws.com/doc/2006-03-01/"><d:Object><d:Key>missing</d:Key></d:Object></d:Delete>'
        )
        self.assert_error(response, 400, "InvalidArgument")

    def test_xml_versioning_and_soft_delete(self):
        bucket = rest_server.db.get_bucket("bucket-name", None)
        for versioning in (False, True):
            for soft_delete in (False, True):
                with self.subTest(versioning=versioning, soft_delete=soft_delete):
                    bucket.metadata.versioning.enabled = versioning
                    bucket.metadata.ClearField("soft_delete_policy")
                    if soft_delete:
                        bucket.metadata.soft_delete_policy.retention_duration.seconds = (
                            7 * 86400
                        )
                    generation = self.put("object")
                    self.result(self.post(self.body([("object", None)])))
                    self.assertEqual(self.get("object").status_code, 404)
                    self.assertEqual(
                        self.get("object", generation=generation).status_code,
                        200 if versioning else 404,
                    )
                    if versioning:
                        self.result(self.post(self.body([("object", generation)])))
                        self.assertEqual(
                            self.get("object", generation=generation).status_code, 404
                        )
                    if soft_delete:
                        response = self.get(
                            "object", generation=generation, softDeleted="true"
                        )
                        self.assertEqual(response.status_code, 200, response.data)
                        self.assertIn("softDeleteTime", response.json)

    def test_held_objects_fail_independently(self):
        for field in ("temporaryHold", "eventBasedHold"):
            for quiet in (None, "True"):
                with self.subTest(field=field, quiet=quiet):
                    generation = self.put("held")
                    response = self.client.patch(
                        "/storage/v1/b/bucket-name/o/held", json={field: True}
                    )
                    self.assertEqual(response.status_code, 200, response.data)
                    self.put("success")
                    result = self.result(
                        self.post(
                            self.body([("held", generation), ("success", None)], quiet)
                        )
                    )
                    error = result.find(NS + "Error")
                    self.assertIsNotNone(error)
                    self.assertEqual(
                        error.findtext(NS + "Code"), "ObjectUnderActiveHold"
                    )
                    self.assertEqual(error.findtext(NS + "Generation"), generation)
                    self.assertEqual(error.findtext(NS + "VersionId"), generation)
                    self.assertEqual(self.get("held").status_code, 200)
                    self.assertEqual(self.get("success").status_code, 404)
                    self.assertEqual(
                        len(result.findall(NS + "Deleted")), 0 if quiet else 1
                    )

    def test_addressing_and_query_parameters(self):
        for path, base_url in (
            ("/bucket-name?delete", None),
            ("/bucket-name/?delete=&userProject=test", None),
            ("/bucket-name?userProject=test&delete", None),
            ("/?delete", "https://bucket-name.storage.googleapis.com"),
            (
                "/?userProject=test&delete=",
                "https://bucket-name.storage.googleapis.com",
            ),
        ):
            with self.subTest(path=path, base_url=base_url):
                self.put("object")
                kwargs = {"base_url": base_url} if base_url else {}
                result = self.result(
                    self.post(self.body([("object", None)]), path, **kwargs)
                )
                self.assertEqual(
                    result.findtext(NS + "Deleted/" + NS + "Key"), "object"
                )
                self.assertEqual(self.get("object").status_code, 404)

    def test_unrelated_routes(self):
        body = self.body([("object", None)])
        self.put("object")
        for path in ("/bucket-name", "/bucket-name/?uploads", "/bucket-name?notdelete"):
            self.assertEqual(self.post(body, path).status_code, 404)
        response = self.post(
            body, "/?uploads", base_url="https://bucket-name.storage.googleapis.com"
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.client.get("/").data, b"OK")
        self.assertEqual(self.post(body, "/bucket-name/object?delete").status_code, 501)
        self.assertEqual(self.get("object").status_code, 200)

    def test_keys_are_xml_text(self):
        names = [
            "internal spaces",
            "raw//a &+%<猫>.txt",
            "raw%2Fencoded.txt",
            "plus+sign",
            "quotes\"'",
            "a\tb\nc",
            "carriage\rreturn",
        ]
        for name in names:
            self.put(name)
        result = self.result(self.post(self.body([(name, None) for name in names])))
        self.assertCountEqual([e.findtext(NS + "Key") for e in result], names)
        for name in names:
            self.assertEqual(self.get(name).status_code, 404, name)

    def test_existing_missing_and_duplicate_targets(self):
        self.put("existing")
        entries = [("existing", None), ("missing", None), ("existing", None)]
        result = self.result(self.post(self.body(entries)))
        self.assertEqual(len(result), 2)
        self.assertCountEqual(
            [e.findtext(NS + "Key") for e in result], ["existing", "missing"]
        )
        self.assertTrue(all(e.tag == NS + "Deleted" for e in result))
        self.assertEqual(self.get("existing").status_code, 404)

    def test_key_whitespace_matches_gcs(self):
        self.put("  object  ")
        self.put("object")
        result = self.result(self.post(self.body([("  object  ", None)])))
        self.assertEqual(result[0].findtext(NS + "Key"), "object")
        self.assertEqual(self.get("object").status_code, 404)
        self.assertEqual(self.get("  object  ").status_code, 200)
        for key in ("", " ", "\t\r\n"):
            with self.subTest(key=key):
                self.put("object")
                self.assert_error(
                    self.post(self.body([("object", None), (key, None)])),
                    400,
                    "InvalidObjectName",
                )
                self.assertEqual(self.get("object").status_code, 200)

    def test_zero_generation_removes_live_version(self):
        bucket = rest_server.db.get_bucket("bucket-name", None)
        bucket.metadata.versioning.enabled = True
        for version in ("0", "00", "+0", "-0"):
            with self.subTest(version=version):
                generation = self.put("object")
                result = self.result(self.post(self.body([("object", version)])))
                self.assertEqual(result[0].findtext(NS + "Generation"), generation)
                self.assertEqual(result[0].findtext(NS + "VersionId"), generation)
                self.assertEqual(self.get("object").status_code, 404)
                self.assertEqual(
                    self.get("object", generation=generation).status_code, 404
                )
        result = self.result(self.post(self.body([("object", "0")])))
        self.assertEqual(result[0].findtext(NS + "VersionId"), "0")

    def test_normalized_duplicate_versions(self):
        result = self.result(
            self.post(
                self.body(
                    [
                        ("missing", "1"),
                        ("missing", "01"),
                        ("missing", "+1"),
                        ("missing", " 1 "),
                        ("missing", "2"),
                    ]
                )
            )
        )
        self.assertCountEqual(
            [e.findtext(NS + "VersionId") for e in result], ["1", "2"]
        )

    def test_zero_generation_preserves_concurrent_replacement(self):
        rest_server.db.get_bucket(
            "bucket-name", None
        ).metadata.versioning.enabled = True
        old = self.put("object", b"old")
        delete = rest_server.db.delete_object

        def replace_then_delete(bucket, key, **kwargs):
            self.put(key, b"replacement")
            return delete(bucket, key, **kwargs)

        with unittest.mock.patch.object(
            rest_server.db, "delete_object", side_effect=replace_then_delete
        ):
            result = self.result(self.post(self.body([("object", "0")])))
        self.assertEqual(result[0].findtext(NS + "VersionId"), old)
        response = self.get("object", alt="media")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"replacement")

    def test_maximum_batch(self):
        entries = [("object-%d" % i, None) for i in range(1000)]
        self.put("object-0")
        self.put("object-999")
        result = self.result(self.post(self.body(entries)))
        self.assertEqual(len(result), 1000)
        self.assertEqual(self.get("object-0").status_code, 404)
        self.assertEqual(self.get("object-999").status_code, 404)

    def test_explicit_generations_preserve_replacement(self):
        bucket = rest_server.db.get_bucket("bucket-name", None)
        bucket.metadata.versioning.enabled = True
        old = self.put("object", b"old")
        new = self.put("object", b"replacement")
        result = self.result(self.post(self.body([("object", old), ("object", "1")])))
        self.assertCountEqual(
            [e.findtext(NS + "VersionId") for e in result], [old, "1"]
        )
        self.assertEqual(self.get("object", generation=old).status_code, 404)
        self.assertEqual(self.get("object").json["generation"], new)
        self.result(self.post(self.body([("object", old)])))
        self.assertEqual(self.get("object").json["generation"], new)
        self.result(self.post(self.body([("object", new)])))
        self.assertEqual(self.get("object").status_code, 404)

    def test_mixed_errors_and_quiet(self):
        for quiet in (None, "True"):
            with self.subTest(quiet=quiet):
                for name in ("success", "denied", "transient"):
                    self.put(name)
                delete = rest_server.db.delete_object

                def fail_selected(bucket, key, **kwargs):
                    if key == "denied":
                        raise RestException("Access denied <&>", 403)
                    if key == "transient":
                        raise RestException("Try again", 503)
                    return delete(bucket, key, **kwargs)

                with unittest.mock.patch.object(
                    rest_server.db, "delete_object", side_effect=fail_selected
                ):
                    result = self.result(
                        self.post(
                            self.body(
                                [
                                    (k, None)
                                    for k in (
                                        "denied",
                                        "success",
                                        "transient",
                                        "missing",
                                    )
                                ],
                                quiet,
                            )
                        )
                    )
                errors = {
                    e.findtext(NS + "Key"): e.findtext(NS + "Code")
                    for e in result.findall(NS + "Error")
                }
                self.assertEqual(
                    errors,
                    {"denied": "AccessDenied", "transient": "SlowDown"},
                )
                self.assertEqual(len(result.findall(NS + "Deleted")), 0 if quiet else 2)
                self.assertEqual(self.get("success").status_code, 404)
                for key in errors:
                    self.assertEqual(self.get(key).status_code, 200)

    def test_whole_request_retry_fault(self):
        self.put("object")
        retry = self.client.post(
            "/retry_test",
            json={"instructions": {"storage.objects.delete": ["return-503"]}},
        ).json
        body = self.body([("object", None)])
        headers = {
            "Content-MD5": base64.b64encode(hashlib.md5(body).digest()).decode(),
            "x-retry-test-id": retry["id"],
        }
        response = self.post(body, headers=headers)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.mimetype, "application/xml")
        self.assertEqual(ET.fromstring(response.data).findtext("Code"), "SlowDown")
        self.assertEqual(self.get("object").status_code, 200)
        self.result(self.post(body, headers=headers))
        self.assertEqual(self.get("object").status_code, 404)

    def test_shared_grpc_database(self):
        port, server = grpc_server.run(0, rest_server.db)
        self.addCleanup(lambda: server.stop(grace=0).wait())
        channel = grpc.insecure_channel("localhost:%d" % port)
        self.addCleanup(channel.close)
        stub = storage_pb2_grpc.StorageStub(channel)
        bucket = "projects/_/buckets/bucket-name"
        rest_server.db.get_bucket(
            "bucket-name", None
        ).metadata.versioning.enabled = True

        def put_grpc(name, content):
            response = stub.WriteObject(
                iter(
                    [
                        storage_pb2.WriteObjectRequest(
                            write_object_spec=storage_pb2.WriteObjectSpec(
                                resource=storage_pb2.Object(bucket=bucket, name=name)
                            ),
                            checksummed_data=storage_pb2.ChecksummedData(
                                content=content
                            ),
                            finish_write=True,
                        )
                    ]
                ),
                timeout=10,
            )
            return response.resource.generation

        generation = put_grpc("grpc-object", b"from gRPC")
        old = put_grpc("replacement", b"old")
        new = put_grpc("replacement", b"new")
        self.put("json-object")
        self.result(
            self.post(
                self.body(
                    [
                        ("grpc-object", str(generation)),
                        ("json-object", None),
                        ("replacement", str(old)),
                    ]
                )
            )
        )
        self.assertEqual(self.get("json-object").status_code, 404)
        with self.assertRaises(grpc.RpcError) as error:
            stub.GetObject(
                storage_pb2.GetObjectRequest(
                    bucket=bucket, object="grpc-object", generation=generation
                ),
                timeout=10,
            )
        self.assertEqual(error.exception.code(), grpc.StatusCode.NOT_FOUND)
        replacement = stub.GetObject(
            storage_pb2.GetObjectRequest(bucket=bucket, object="replacement"),
            timeout=10,
        )
        self.assertEqual(replacement.generation, new)
