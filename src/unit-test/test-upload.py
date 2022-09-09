from gladia_api_utils.OVHFileManager import OVHFileManager

file_manager = OVHFileManager()
file_manager.upload_file_from_path('unit-test/test.png', 'test/test/COUCOU.png')