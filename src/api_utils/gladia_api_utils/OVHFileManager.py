import os
from logging import getLogger

import swiftclient

logger = getLogger(__name__)


class OVHFileManager:
    def __init__(self):
        self.USERNAME = os.getenv("OVH_OBJECT_STORAGE_USERNAME")
        self.KEY = os.getenv("OVH_OBJECT_STORAGE_KEY")
        self.AUTH_URL = os.getenv("OVH_OBJECT_STORAGE_AUTH_URL")
        self.CONTAINER_NAME = os.getenv("OVH_OBJECT_STORAGE_CONTAINER_NAME")
        self.REGION_NAME = os.getenv("OVH_OBJECT_STORAGE_REGION_NAME")
        self.TENANT_NAME = os.getenv("OVH_OBJECT_STORAGE_TENANT_NAME")
        self.conn = self.get_connexion()
        self.content_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".png": "image/png",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/x-m4a",
        }

    def get_connexion(self):
        conn = swiftclient.Connection(
            user=self.USERNAME,
            key=self.KEY,
            authurl=self.AUTH_URL,
            tenant_name=self.TENANT_NAME,
            auth_version=self.AUTH_URL[-1:],
            os_options={
                "region_name": self.REGION_NAME,
            },
        )
        return conn

    def get_objects(self, prefix="examples/"):
        raw_result = self.conn.get_container(self.CONTAINER_NAME, prefix=prefix)[1]
        list_objects = [res["name"] for res in raw_result]
        return list_objects

    def upload_file_from_path(self, file_path, ovh_file_name):
        file_extension = os.path.splitext(file_path)[-1]
        with open(file_path, "rb") as file:
            try:
                self.conn.put_object(
                    self.CONTAINER_NAME,
                    ovh_file_name,
                    contents=file.read(),
                    content_type=self.content_type[file_extension],
                )
            except Exception as e:
                logger.error(f"Error uploading file on ovh {file_path}")
                logger.error(f"Error message: {e}")

    def delete_file(self, ovh_file_name):
        try:
            self.conn.delete_object(self.CONTAINER_NAME, ovh_file_name)
        except Exception as e:
            logger.error(f"Error deleting file on ovh {ovh_file_name}")
            logger.error(f"Error message {e}")
