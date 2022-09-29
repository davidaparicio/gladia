import io
import os
import random
import re
import string
import sys
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Any, Tuple

import gdown
import magic
import requests
from PIL import Image
from xtract import XZ, BZip2, GZip, Rar, Tar, Zip, xtract
from xtract.utils import get_file_type

logger = getLogger(__name__)

PATTERN = re.compile(r'((\w:)|(\.))((/(?!/)(?!/)|\\{2})[^\n?"|></\\:*]+)+')


class PathNotExistsException(Exception):
    pass


class FileProcessingError(Exception):
    pass


MIME_TYPES_CATEGORIES = {
    "audio": "audio",
    "video": "video",
    "image": "image",
    "text": "text",
    "font": "font",
    "binary": "binary",
    "application": "application",
    "flat_structured_data": "flat_structured_data",
    "multidimensional_structured_data": "multidimensional_structured_data",
    "ebook": "ebook",
    "unknown": "unknown",
    "archive": "archive",
    "executable": "executable",
    "web_content": "web_content",
    "document": "document",
    "spreadsheet": "spreadsheet",
    "presentation": "presentation",
    "database": "database",
    "pdf": "pdf",
    "diagram": "diagram",
    "calendar": "calendar",
}

MIME_TYPE_TO_CATEGORY = {
    "audio/aac": {
        "category": MIME_TYPES_CATEGORIES["audio"],
        "description": "AAC audio",
        "extension": "aac",
    },
    "audio/midi": {
        "category": MIME_TYPES_CATEGORIES["audio"],
        "description": "Musical Instrument Digital Interface (MIDI)",
        "extension": "mid",
    },
    "audio/ogg": {
        "category": MIME_TYPES_CATEGORIES["audio"],
        "description": "OGG audio",
        "extension": "ogg",
    },
    "audio/x-wav": {
        "category": MIME_TYPES_CATEGORIES["audio"],
        "description": "Waveform Audio Format",
        "extension": "wav",
    },
    "audio/webm": {
        "category": MIME_TYPES_CATEGORIES["audio"],
        "description": "WEBM audio",
        "extension": "webm",
    },
    "audio/3gpp": {
        "category": MIME_TYPES_CATEGORIES["audio"],
        "description": "3GPP audio",
        "extension": "3gp",
    },
    "audio/3gpp2": {
        "category": MIME_TYPES_CATEGORIES["audio"],
        "description": "3GPP2 audio",
        "extension": "3g2",
    },
    "video/x-msvideo": {
        "category": MIME_TYPES_CATEGORIES["video"],
        "description": "AVI: Audio Video Interleave",
        "extension": "avi",
    },
    "video/mpeg": {
        "category": MIME_TYPES_CATEGORIES["video"],
        "description": "MPEG Video",
        "extension": "mpeg",
    },
    "video/ogg": {
        "category": MIME_TYPES_CATEGORIES["video"],
        "description": "OGG video",
        "extension": "ogv",
    },
    "video/webm": {
        "category": MIME_TYPES_CATEGORIES["video"],
        "description": "WEBM video",
        "extension": "webm",
    },
    "video/3gpp": {
        "category": MIME_TYPES_CATEGORIES["video"],
        "description": "3GPP video",
        "extension": "3gp",
    },
    "video/3gpp2": {
        "category": MIME_TYPES_CATEGORIES["video"],
        "description": "3GPP2 video",
        "extension": "3g2",
    },
    "image/bmp": {
        "category": MIME_TYPES_CATEGORIES["image"],
        "description": "Windows OS/2 Bitmap Graphics",
        "extension": "bmp",
    },
    "image/gif": {
        "category": MIME_TYPES_CATEGORIES["image"],
        "description": "Graphics Interchange Format (GIF)",
        "extension": "gif",
    },
    "image/x-icon": {
        "category": MIME_TYPES_CATEGORIES["image"],
        "description": "Icon format",
        "extension": "ico",
    },
    "image/jpeg": {
        "category": MIME_TYPES_CATEGORIES["image"],
        "description": "JPEG images",
        "extension": "jpg",
    },
    "image/png": {
        "category": MIME_TYPES_CATEGORIES["image"],
        "description": "Portable Network Graphics",
        "extension": "png",
    },
    "image/svg+xml": {
        "category": MIME_TYPES_CATEGORIES["image"],
        "description": "Scalable Vector Graphics (SVG)",
        "extension": "svg",
    },
    "image/tiff": {
        "category": MIME_TYPES_CATEGORIES["image"],
        "description": "Tagged Image File Format (TIFF)",
        "extension": "tiff",
    },
    "image/webp": {
        "category": MIME_TYPES_CATEGORIES["image"],
        "description": "WEBP image",
        "extension": "webp",
    },
    "font/otf": {
        "category": MIME_TYPES_CATEGORIES["font"],
        "description": "OpenType font",
        "extension": "otf",
    },
    "font/ttf": {
        "category": MIME_TYPES_CATEGORIES["font"],
        "description": "TrueType Font",
        "extension": "ttf",
    },
    "font/woff": {
        "category": MIME_TYPES_CATEGORIES["font"],
        "description": "Web Open Font Format (WOFF)",
        "extension": "woff",
    },
    "font/woff2": {
        "category": MIME_TYPES_CATEGORIES["font"],
        "description": "Web Open Font Format (WOFF)",
        "extension": "woff2",
    },
    "application/octet-stream": {
        "category": MIME_TYPES_CATEGORIES["binary"],
        "description": "Any kind of binary data",
        "extension": "bin",
    },
    "application/vnd.amazon.ebook": {
        "category": MIME_TYPES_CATEGORIES["ebook"],
        "description": "Amazon Kindle eBook format",
        "extension": "azw",
    },
    "application/epub+zip": {
        "category": MIME_TYPES_CATEGORIES["ebook"],
        "description": "Electronic publication (EPUB)",
        "extension": "epub",
    },
    "application/x-bzip": {
        "category": MIME_TYPES_CATEGORIES["archive"],
        "description": "BZip archive",
        "extension": "bz",
    },
    "application/x-bzip2": {
        "category": MIME_TYPES_CATEGORIES["archive"],
        "description": "BZip2 archive",
        "extension": "bz2",
    },
    "application/java-archive": {
        "category": MIME_TYPES_CATEGORIES["archive"],
        "description": "Java Archive (JAR)",
        "extension": "jar",
    },
    "application/x-rar-compressed": {
        "category": MIME_TYPES_CATEGORIES["archive"],
        "description": "RAR archive",
        "extension": "rar",
    },
    "application/x-tar": {
        "category": MIME_TYPES_CATEGORIES["archive"],
        "description": "Tape Archive (TAR)",
        "extension": "tar",
    },
    "application/zip": {
        "category": MIME_TYPES_CATEGORIES["archive"],
        "description": "ZIP archive",
        "extension": "zip",
    },
    "application/x-7z-compressed": {
        "category": MIME_TYPES_CATEGORIES["archive"],
        "description": "7-zip archive",
        "extension": "7z",
    },
    "application/x-csh": {
        "category": MIME_TYPES_CATEGORIES["executable"],
        "description": "C-Shell script",
        "extension": "csh",
    },
    "application/ogg": {
        "category": MIME_TYPES_CATEGORIES["executable"],
        "description": "OGG",
        "extension": "ogx",
    },
    "application/x-sh": {
        "category": MIME_TYPES_CATEGORIES["executable"],
        "description": "Bourne shell script",
        "extension": "sh",
    },
    "application/x-shockwave-flash": {
        "category": MIME_TYPES_CATEGORIES["executable"],
        "description": "Adobe Flash",
        "extension": "swf",
    },
    "text/css": {
        "category": MIME_TYPES_CATEGORIES["web_content"],
        "description": "Cascading Style Sheets (CSS)",
        "extension": "css",
    },
    "text/html": {
        "category": MIME_TYPES_CATEGORIES["web_content"],
        "description": "HyperText Markup Language (HTML)",
        "extension": "html",
    },
    "application/javascript": {
        "category": MIME_TYPES_CATEGORIES["web_content"],
        "description": "JavaScript",
        "extension": "js",
    },
    "application/xhtml+xml": {
        "category": MIME_TYPES_CATEGORIES["web_content"],
        "description": "XHTML",
        "extension": "xhtml",
    },
    "application/vnd.mozilla.xul+xml": {
        "category": MIME_TYPES_CATEGORIES["web_content"],
        "description": "XUL",
        "extension": "xul",
    },
    "text/csv": {
        "category": MIME_TYPES_CATEGORIES["flat_structured_data"],
        "description": "Comma-separated values (CSV)",
        "extension": "csv",
    },
    "text/tab-separated-values": {
        "category": MIME_TYPES_CATEGORIES["flat_structured_data"],
        "description": "Tab-separated values (TSV)",
        "extension": "tsv",
    },
    "application/vnd.ms-excel": {
        "category": MIME_TYPES_CATEGORIES["spreadsheet"],
        "description": "Microsoft Excel",
        "extension": "xls",
    },
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {
        "category": MIME_TYPES_CATEGORIES["spreadsheet"],
        "description": "Microsoft Excel (OpenXML)",
        "extension": "xlsx",
    },
    "application/vnd.openxmlformats-officedocument.spreadsheetml.template": {
        "category": MIME_TYPES_CATEGORIES["spreadsheet"],
        "description": "Microsoft Excel (OpenXML)",
        "extension": "xltx",
    },
    "application/vnd.ms-excel.sheet.macroEnabled.12": {
        "category": MIME_TYPES_CATEGORIES["spreadsheet"],
        "description": "Microsoft Excel (OpenXML)",
        "extension": "xlsm",
    },
    "application/vnd.ms-excel.template.macroEnabled.12": {
        "category": MIME_TYPES_CATEGORIES["spreadsheet"],
        "description": "Microsoft Excel (OpenXML)",
        "extension": "xltm",
    },
    "application/vnd.ms-excel.addin.macroEnabled.12": {
        "category": MIME_TYPES_CATEGORIES["spreadsheet"],
        "description": "Microsoft Excel (OpenXML)",
        "extension": "xlam",
    },
    "application/vnd.ms-excel.sheet.binary.macroEnabled.12": {
        "category": MIME_TYPES_CATEGORIES["spreadsheet"],
        "description": "Microsoft Excel (OpenXML)",
        "extension": "xlsb",
    },
    "application/vnd.oasis.opendocument.spreadsheet": {
        "category": MIME_TYPES_CATEGORIES["spreadsheet"],
        "description": "OpenDocument spreadsheet",
        "extension": "ods",
    },
    "application/vnd.oasis.opendocument.spreadsheet-template": {
        "category": MIME_TYPES_CATEGORIES["spreadsheet"],
        "description": "OpenDocument spreadsheet template",
        "extension": "ots",
    },
    "application/vnd.oasis.opendocument.spreadsheet-flat-xml": {
        "category": MIME_TYPES_CATEGORIES["spreadsheet"],
        "description": "OpenDocument spreadsheet (Flat XML)",
        "extension": "fods",
    },
    "application/vnd.lotus-1-2-3": {
        "category": MIME_TYPES_CATEGORIES["spreadsheet"],
        "description": "Lotus 1-2-3",
        "extension": "123",
    },
    "application/vnd.lotus-approach": {
        "category": MIME_TYPES_CATEGORIES["spreadsheet"],
        "description": "Lotus Approach",
        "extension": "apr",
    },
    "application/vnd.lotus-freelance": {
        "category": MIME_TYPES_CATEGORIES["spreadsheet"],
        "description": "Lotus Freelance",
        "extension": "pre",
    },
    "application/xslt+xml": {
        "category": MIME_TYPES_CATEGORIES["spreadsheet"],
        "description": "XSLT",
        "extension": "xslt",
    },
    "application/json": {
        "category": MIME_TYPES_CATEGORIES["multidimensional_structured_data"],
        "description": "JSON",
        "extension": "json",
    },
    "application/xml": {
        "category": MIME_TYPES_CATEGORIES["multidimensional_structured_data"],
        "description": "XML",
        "extension": "xml",
    },
    "application/msword": {
        "category": MIME_TYPES_CATEGORIES["document"],
        "description": "Microsoft Word",
        "extension": "doc",
    },
    "application/x-abiword": {
        "category": MIME_TYPES_CATEGORIES["document"],
        "description": "AbiWord",
        "extension": "abw",
    },
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {
        "category": MIME_TYPES_CATEGORIES["document"],
        "description": "Microsoft Word (OpenXML)",
        "extension": "docx",
    },
    "application/vnd.oasis.opendocument.text": {
        "category": MIME_TYPES_CATEGORIES["document"],
        "description": "OpenDocument text",
        "extension": "odt",
    },
    "application/rtf": {
        "category": MIME_TYPES_CATEGORIES["document"],
        "description": "Rich Text Format (RTF)",
        "extension": "rtf",
    },
    "application/vnd.ms-powerpoint": {
        "category": MIME_TYPES_CATEGORIES["presentation"],
        "description": "Microsoft PowerPoint",
        "extension": "ppt",
    },
    "application/vnd.oasis.opendocument.presentation": {
        "category": MIME_TYPES_CATEGORIES["presentation"],
        "description": "OpenDocument presentation",
        "extension": "odp",
    },
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": {
        "category": MIME_TYPES_CATEGORIES["presentation"],
        "description": "Microsoft PowerPoint (OpenXML)",
        "extension": "pptx",
    },
    "application/pdf": {
        "category": MIME_TYPES_CATEGORIES["document"],
        "description": "Adobe Portable Document Format (PDF)",
        "extension": "pdf",
    },
    "application/vnd.visio": {
        "category": MIME_TYPES_CATEGORIES["diagram"],
        "description": "Microsoft Visio",
        "extension": "vsd",
    },
    "application/vnd.oasis.opendocument.graphics": {
        "category": MIME_TYPES_CATEGORIES["diagram"],
        "description": "OpenDocument drawing",
        "extension": "odg",
    },
    "application/vnd.openxmlformats-officedocument.drawingml.diagramData+xml": {
        "category": MIME_TYPES_CATEGORIES["diagram"],
        "description": "Microsoft Visio (OpenXML)",
        "extension": "vsdx",
    },
    "application/vnd.openxmlformats-officedocument.drawingml.diagramColors+xml": {
        "category": MIME_TYPES_CATEGORIES["diagram"],
        "description": "Microsoft Visio (OpenXML)",
        "extension": "vssx",
    },
    "application/vnd.openxmlformats-officedocument.drawingml.diagramLayout+xml": {
        "category": MIME_TYPES_CATEGORIES["diagram"],
        "description": "Microsoft Visio (OpenXML)",
        "extension": "vstx",
    },
    "application/vnd.openxmlformats-officedocument.drawingml.diagramStyle+xml": {
        "category": MIME_TYPES_CATEGORIES["diagram"],
        "description": "Microsoft Visio (OpenXML)",
        "extension": "vssx",
    },
    "application/vnd.ms-office.drawingml.diagramData+xml": {
        "category": MIME_TYPES_CATEGORIES["diagram"],
        "description": "Microsoft Visio (OpenXML)",
        "extension": "vsdx",
    },
    "application/vnd.ms-office.drawingml.diagramColors+xml": {
        "category": MIME_TYPES_CATEGORIES["diagram"],
        "description": "Microsoft Visio (OpenXML)",
        "extension": "vssx",
    },
    "application/vnd.ms-office.drawingml.diagramLayout+xml": {
        "category": MIME_TYPES_CATEGORIES["diagram"],
        "description": "Microsoft Visio (OpenXML)",
        "extension": "vstx",
    },
    "application/vnd.ms-office.drawingml.diagramStyle+xml": {
        "category": MIME_TYPES_CATEGORIES["diagram"],
        "description": "Microsoft Visio (OpenXML)",
        "extension": "vssx",
    },
    "text/plain": {
        "category": MIME_TYPES_CATEGORIES["text"],
        "description": "Plain text",
        "extension": "txt",
    },
    "text/calendar": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "iCalendar",
        "extension": "ics",
    },
    "text/x-vcard": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vCard",
        "extension": "vcf",
    },
    "text/x-vcalendar": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vCalendar",
        "extension": "vcs",
    },
    "text/x-vnote": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vNote",
        "extension": "vnt",
    },
    "text/x-vtodo": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vTodo",
        "extension": "vtd",
    },
    "text/x-vjournal": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vJournal",
        "extension": "vjd",
    },
    "text/x-vmessage": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vMessage",
        "extension": "vmg",
    },
    "text/x-vfreebusy": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vFreeBusy",
        "extension": "vfb",
    },
    "text/x-vtimezone": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vTimezone",
        "extension": "vtz",
    },
    "text/x-vavailability": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vAvailability",
        "extension": "vav",
    },
    "text/x-vpoll": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vPoll",
        "extension": "vpl",
    },
    "text/x-vdirectory": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vDirectory",
        "extension": "vdr",
    },
    "text/x-vconference": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vConference",
        "extension": "vcf",
    },
    "text/x-vexample": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vExample",
        "extension": "vex",
    },
    "text/x-vrsvp": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vRSVP",
        "extension": "vrs",
    },
    "text/x-vrsvp-reply": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vRSVP-Reply",
        "extension": "vrr",
    },
    "text/x-vrsvp-request": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vRSVP-Request",
        "extension": "vrr",
    },
    "text/x-vrsvp-error": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vRSVP-Error",
        "extension": "vre",
    },
    "text/x-vrsvp-reminder": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vRSVP-Reminder",
        "extension": "vrm",
    },
    "text/x-vrsvp-availability": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vRSVP-Availability",
        "extension": "vra",
    },
    "text/x-vrsvp-poll": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vRSVP-Poll",
        "extension": "vrp",
    },
    "text/x-vrsvp-poll-change": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vRSVP-Poll-Change",
        "extension": "vpc",
    },
    "text/x-vrsvp-poll-vote": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vRSVP-Poll-Vote",
        "extension": "vpv",
    },
    "text/x-vrsvp-poll-published": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vRSVP-Poll-Published",
        "extension": "vpp",
    },
    "text/x-vrsvp-poll-expired": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vRSVP-Poll-Expired",
        "extension": "vpe",
    },
    "text/x-vrsvp-poll-closed": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vRSVP-Poll-Closed",
        "extension": "vpc",
    },
    "text/x-vrsvp-poll-answered": {
        "category": MIME_TYPES_CATEGORIES["calendar"],
        "description": "vRSVP",
        "extension": "vpa",
    },
}


def is_binary_file(file_path: str) -> bool:
    """
    Check if a file is binary or not.

    Args:
        file_path (str): The path to the file to check.

    Returns:
        bool: True if the file is binary, False otherwise.
    """

    textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})
    is_binary_string = lambda bytes: bool(bytes.translate(None, textchars))

    return is_binary_string(open(file_path, "rb").read(1024))


def is_valid_path(string: str) -> bool:
    """
    Check if a string is a valid path.

    Args:
        string (str): The string to check.

    Returns:
        bool: True if the string is a valid path, False otherwise.
    """

    if string and isinstance(string, str) and PATTERN.match(string):
        return True
    else:
        return False


def get_tmp_filename() -> str:
    """
    Get a random temporary fullpath.

    Returns:
        str: The random filepath.
    """
    return os.path.join(
        tempfile._get_default_tempdir(), next(tempfile._get_candidate_names())
    )


def write_tmp_file(content: Any) -> str:
    """
    Write content to a temporary file.

    Args:
        content (Any): The content to write.

    Returns:
        str: The path to the temporary file.
    """

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(content)
    tmp.close()

    return tmp.name


def input_to_files(func):
    """
    Decorator that converts input to a list of files.

    Args:
        func (function): The function to decorate.

    Returns:
        function: The decorated function.
    """

    def inner(*args, **kwargs) -> Any:
        """
        Wrapper for the decorated function.

        Args:
            *args: The arguments to the function.
            **kwargs: The keyword arguments to the function.

        Returns:
            Any: The return value of the function.
        """
        tmp_files = list()

        for keyword, arg in kwargs.items():
            if isinstance(arg, (bytes, bytearray)):
                tmp_file = write_tmp_file(arg)
                tmp_files.append(tmp_file)
                kwargs[keyword] = tmp_file

        result = func(*args, **kwargs)

        try:
            delete_all_files(tmp_files)
        except Exception as e:
            logger.error(f"Couldn't delete tmp files: {e}")

        return result

    return inner


def download_file(
    url: str,
    file_full_path: str,
    force_create_dir: bool = True,
    force_redownload: bool = False,
) -> Path:
    """
    Download a file from a url.

    Args:
        url (str): The url to download the file from.
        file_full_path (str): The path to the file to download.
        force_create_dir (bool): Create the directory if it doesn't exist. (default: True)
        force_redownload (bool): Redownload the file if it already exists. (default: False)

    Returns:
        Path: The path to the downloaded file.
    """

    file_full_path = Path(file_full_path)

    if not os.path.exists(file_full_path.parent):
        if force_create_dir:
            create_directory(file_full_path.parent)
        else:
            raise PathNotExistsException(
                "Parent directory doesn't exists : change the path or use force_create_dir = True"
            )

    if not file_full_path.exists() or force_redownload:
        write_url_content_to_file(file_full_path, url)

    return file_full_path


def write_url_content_to_file(file_full_path: Path, url: str) -> bool:
    """
    Write the content of a url to a file.

    Args:
        file_full_path (Path): The path to the file to write.
        url (str): The url to download the content from.

    Returns:
        bool: True if the file was written, False otherwise.
    """
    if url.startswith("https://drive.google.com/"):
        gdown.download(url, file_full_path, quiet=False)
        return True
    else:
        data = requests.get(url).content

        logger.debug(f"writing {url} to {file_full_path}")

        return write_to_file(file_full_path, data)


def write_to_file(file_full_path: str, data: Any, overwrite: bool = False) -> bool:
    """
    Write data to a file.

    Args:
        file_full_path (str): The path to the file to write.
        data (Any): The data to write.
        overwrite (bool): Overwrite the file if it already exists. (default: False)

    Returns:
        bool: True if the file was written, False otherwise.
    """

    if isinstance(file_full_path, str):
        file_full_path = Path(file_full_path)

    os.makedirs(os.path.dirname(file_full_path), exist_ok=True)

    if not os.path.exists(file_full_path) or overwrite:
        if isinstance(data, io.BytesIO):
            with open(file_full_path, "wb") as handler:
                data = data.read()
            handler.write(data)
        elif isinstance(data, Image.Image):
            data.save(file_full_path)
        else:
            with open(file_full_path, "wb") as handler:
                handler.write(data)


def delete_file(filepath: str) -> bool:
    """
    Delete a file.

    Args:
        filepath (str): The path to the file to delete.

    Returns:
        bool: True if the file was deleted, False otherwise.
    """

    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            return True
        except Exception as e:
            logger.error(f"Couldn't delete file {filepath}: {e}")
            return False
    else:
        return False


def delete_all_files(files: list) -> dict:
    """
    Delete all files in a given list

    Args:
        files (list): The list of files to delete.

    Returns:
        list: The list of files that were deleted.
    """

    out = dict()
    for file in files:
        out[file] = delete_file(file)
    return out


def delete_directory(dirpath: str) -> bool:
    """
    Delete a file.

    Args:
        dirpath (str): The path to the directory to delete.

    Returns:
        bool: True if the directory was deleted, False otherwise.
    """

    delete_file(dirpath)


def create_directory(path: str) -> bool:
    """
    Create a directory.

    Args:
        path (str): The path to the directory to create.

    Returns:
        bool: True if the directory was created, False otherwise.
    """

    if len(os.path.dirname(path)) > 0:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Couldn't create directory {path}: {e}")
            return False
    else:
        return False


def uncompress(
    path: str,
    destination: str = None,
    delete_after_uncompress: bool = True,
    force_create_dir: bool = True,
) -> str:
    """
    Uncompress a file.

    Args:
        path (str): The path to the file to uncompress.
        destination (str): The path to the destination directory. If None, the destination is the same as the path. (default: None)
        delete_after_uncompress (bool): Delete the file after uncompress. (default: True)

    Returns:
        str: The path to the uncompressed file.

    Raises:
        Exception: If the file is not a supported compression format.
    """

    try:
        if force_create_dir:
            create_directory(destination)
        logger.debug(f"Extracting archive from {path} to {destination}")
        output = xtract(path, destination=destination, overwrite=True, all=False)

        if delete_after_uncompress:
            logger.debug(f"Deleting archive {path}")
            delete_file(path)

        logger.debug(f"Extraction done to {output}")

        return output

    except:
        raise FileProcessingError(f"Error while uncompressing {path}")


def compress_directory(
    path: str,
    compression_format: str = "gzip",
    destination: str = None,
    delete_after_compress: bool = False,
) -> str:
    """
    Compress a directory and returns the path of the compressed file.

    Args:
        path (str): path to the directory to compress
        compression_format (str): format of the compression (rar, tar, zip, bz2, gz, xz) (default: gzip)
        destination (str): path to the destination of the compressed file. if None, the compressed file is in the same directory as the directory to compress (default: None)
        delete_after_compress (bool): delete the directory after compression (default: False)

    Returns:
        str: path to the compressed file
    """
    output = ""

    if destination:
        # used for relative paths
        namespace = sys._getframe(1).f_globals
        cwd = os.getcwd()
        rel_path = namespace["__file__"]
        root_path = os.path.dirname(os.path.join(cwd, rel_path))

        if not os.path.isabs(destination):
            destination = os.path.join(root_path, destination)

    if compression_format == "rar":
        output = Rar(path, destination=destination)

    if compression_format == "tar":
        output = Tar(path, destination=destination)

    if compression_format == "zip":
        output = Zip(path, destination=destination)

    if compression_format == "bz2":
        output = BZip2(path, destination=destination)

    if compression_format == "gz":
        output = GZip(path, destination=destination)

    if compression_format == "xz":
        output = XZ(path, destination=destination)

    if delete_after_compress:
        logger.debug(f"Deleting directory {path}")
        delete_directory(path)

    return output


def is_uncompressable(file_path: str) -> bool:
    """
    Returns True if the file is an uncompressable file.

    Args:
        file_path (str): The path to the file to analyze.

    Returns:
        bool: True if the file is an uncompressable file.
    """

    # used for relative paths
    namespace = sys._getframe(1).f_globals
    cwd = os.getcwd()
    rel_path = namespace["__file__"]
    root_path = os.path.dirname(os.path.join(cwd, rel_path))

    if not os.path.isabs(file_path):
        file_path = os.path.join(root_path, file_path)

    # exclude archive files that are not supported
    if file_path.endswith(
        (".ckpt", ".pt", ".pth", ".pkl", ".pth", ".pkl", ".ckpt.meta")
    ):
        return False
    else:
        return is_archive(file_path)


def is_archive(file_path: str) -> bool:
    """
    Returns True if the file is an archive file.

    Args:
        file_path (str): The path to the file to analyze.

    Returns:
        bool: True if the file is an archive file.
    """

    # used for relative paths
    namespace = sys._getframe(1).f_globals
    cwd = os.getcwd()
    rel_path = namespace["__file__"]
    root_path = os.path.dirname(os.path.join(cwd, rel_path))

    if not os.path.isabs(file_path):
        file_path = os.path.join(root_path, file_path)

    return get_file_category(file_path) == "archive"


def is_image(file_path: str) -> bool:
    """
    Returns True if the file is an image file.

    Args:
        file_path (str): The path to the file to analyze.

    Returns:
        bool: True if the file is an image file.
    """

    # used for relative paths
    namespace = sys._getframe(1).f_globals
    cwd = os.getcwd()
    rel_path = namespace["__file__"]
    root_path = os.path.dirname(os.path.join(cwd, rel_path))

    if not os.path.isabs(file_path):
        file_path = os.path.join(root_path, file_path)

    file_mime_type = get_file_category(file_path)

    return file_mime_type == "image"


def is_audio(file_path: str) -> bool:
    """
    Returns True if the file is an audio file.

    Args:
        file_path (str): The path to the file to analyze.

    Returns:
        bool: True if the file is an audio file.
    """

    # used for relative paths
    namespace = sys._getframe(1).f_globals
    cwd = os.getcwd()
    rel_path = namespace["__file__"]
    root_path = os.path.dirname(os.path.join(cwd, rel_path))

    if not os.path.isabs(file_path):
        file_path = os.path.join(root_path, file_path)

    file_mime_type = get_file_category(file_path)

    return file_mime_type == "audio"


def is_video(file_path: str) -> bool:
    """
    Returns True if the file is a video file.

    Args:
        file_path (str): The path to the file to analyze.

    Returns:
        bool: True if the file is a video file.
    """

    # used for relative paths
    namespace = sys._getframe(1).f_globals
    cwd = os.getcwd()
    rel_path = namespace["__file__"]
    root_path = os.path.dirname(os.path.join(cwd, rel_path))

    if not os.path.isabs(file_path):
        file_path = os.path.join(root_path, file_path)

    file_mime_type = get_file_category(file_path)

    return file_mime_type == "video"


def is_structured_data(file_path: str) -> bool:
    """
    Returns True if the file is a structured data file.

    Args:
        file_path (str): The path to the file to analyze.

    Returns:
        bool: True if the file is a structured data file.
    """

    # used for relative paths
    namespace = sys._getframe(1).f_globals
    cwd = os.getcwd()
    rel_path = namespace["__file__"]
    root_path = os.path.dirname(os.path.join(cwd, rel_path))

    if not os.path.isabs(file_path):
        file_path = os.path.join(root_path, file_path)

    file_mime_type = get_file_category(file_path)

    return file_mime_type == "structured_data"


def is_word(file_path: str) -> bool:
    """
    Returns True if the file is a word file.

    Args:
        file_path (str): The path to the file to analyze.

    Returns:
        bool: True if the file is a word file.
    """

    # used for relative paths
    namespace = sys._getframe(1).f_globals
    cwd = os.getcwd()
    rel_path = namespace["__file__"]
    root_path = os.path.dirname(os.path.join(cwd, rel_path))

    if not os.path.isabs(file_path):
        file_path = os.path.join(root_path, file_path)

    file_mime_type = get_file_category(file_path)

    return file_mime_type == "word"


def is_pdf(file_path: str) -> bool:
    """
    Returns True if the file is a pdf file.

    Args:
        file_path (str): The path to the file to analyze.

    Returns:
        bool: True if the file is a pdf file.
    """

    # used for relative paths
    namespace = sys._getframe(1).f_globals
    cwd = os.getcwd()
    rel_path = namespace["__file__"]
    root_path = os.path.dirname(os.path.join(cwd, rel_path))

    if not os.path.isabs(file_path):
        file_path = os.path.join(root_path, file_path)

    file_mime_type = get_file_category(file_path)

    return file_mime_type == "pdf"


def is_web_content(file_path: str) -> bool:
    """
    Returns True if the file is a web content file.

    Args:
        file_path (str): The path to the file to analyze.

    Returns:
        bool: True if the file is a web content file.
    """

    # used for relative paths
    namespace = sys._getframe(1).f_globals
    cwd = os.getcwd()
    rel_path = namespace["__file__"]
    root_path = os.path.dirname(os.path.join(cwd, rel_path))

    if not os.path.isabs(file_path):
        file_path = os.path.join(root_path, file_path)

    file_mime_type = get_file_category(file_path)

    return file_mime_type == "web_content"


def get_buffer_category(buffer: bytes) -> str:
    """
    Returns the category of the buffer.

    Args:
        buffer (bytes): The buffer to analyze.

    Returns:
        str: The category of the buffer.
    """

    return get_mime_category(get_buffer_type(buffer))


def get_file_category(file_path: str) -> str:
    """
    Returns the category of the file.

    Args:
        file_path (str): The path to the file to analyze.

    Returns:
        str: The category of the file.
    """

    # used for relative paths
    namespace = sys._getframe(1).f_globals
    cwd = os.getcwd()
    rel_path = namespace["__file__"]
    root_path = os.path.dirname(os.path.join(cwd, rel_path))

    if not os.path.isabs(file_path):
        file_path = os.path.join(root_path, file_path)

    file_mime_type = get_file_type(file_path)

    return get_mime_category(file_mime_type)


def get_mime_category(mime_type: str) -> str:
    """
    Returns the category of the mime type.

    Args:
        mime_type (str): The mime type to analyze.

    Returns:
        str: The category of the mime type.
    """

    if mime_type in MIME_TYPE_TO_CATEGORY:
        return MIME_TYPE_TO_CATEGORY[mime_type]["category"]

    return MIME_TYPES_CATEGORIES["unknown"]


def get_mime_extension(mime_type: str) -> str:
    """
    Returns the category of the mime type.

    Args:
        mime_type (str): The mime type to analyze.

    Returns:
        str: The category of the mime type.
    """

    if mime_type in MIME_TYPE_TO_CATEGORY:
        return MIME_TYPE_TO_CATEGORY[mime_type]["extension"]

    return ""


def get_file_extension(file_path: str) -> str:
    """
    Returns the file extension associated to the mime type

    Args:
        file_path (str): The path to the file to analyze.

    Returns:
        str: the file extension associated to the mime type
    """

    if not os.path.isabs(file_path):
        namespace = sys._getframe(1).f_globals
        cwd = os.getcwd()
        rel_path = namespace["__file__"]
        root_path = os.path.dirname(os.path.join(cwd, rel_path))
        file_path = os.path.join(root_path, file_path)

    file_mime_type = get_file_type(file_path)
    file_extension = get_mime_extension(file_mime_type)

    return file_extension


def get_file_type(file_path: str) -> str:
    """
    Return the file mime type with mime type
    see full list here : https://developer.mozilla.org/fr/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types

    Args:
        file_path (str): The path to the file to analyze.

    Returns:
        str: The mime type of the file.
    """

    # used for relative paths

    if not os.path.isabs(file_path):
        namespace = sys._getframe(1).f_globals
        cwd = os.getcwd()
        rel_path = namespace["__file__"]
        root_path = os.path.dirname(os.path.join(cwd, rel_path))
        file_path = os.path.join(root_path, file_path)

    return magic.from_file(str(file_path), mime=True)


def get_buffer_type(buffer: bytes) -> str:
    """
    Return the buffer mime type with mime type
    see full list here : https://developer.mozilla.org/fr/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Common_types

    Args:
        buffer (bytes): The buffer to analyze.

    Returns:
        str: The mime type of the buffer.
    """

    return magic.from_buffer(buffer, mime=True)


def random_string(lenght: int = 10) -> str:
    """
    Generate a random string composed of lower cased ascii characters

    Args:
        lenght (int): The length of the string to generate. (default: 10)

    Returns:
        str: random string of lenght `lenght` composed of lower cased ascii characters
    """

    letters = string.ascii_lowercase

    generated_random_string = "".join(random.choice(letters) for _ in range(lenght))

    return generated_random_string


def create_random_directory(root_path: str) -> str:
    """
    Create a random directory in the root_path.

    Args:
        root_path (str): The path to the root directory to create the file into.

    Returns:
        str: The path to the created directory.
    """
    full_path = os.path.join(root_path, random_string(10))
    create_directory(full_path)

    return full_path


def generate_random_filename(root_path: str, extension: str) -> Tuple[str, str]:
    """
    Generate a random filename in the root_path.

    Args:
        root_path (str): The path to the root directory to create the file into.
        extension (str): The extension of the file.

    Returns:
        str: The path to the created file.
    """

    filename = ".".join([random_string(10), extension])
    full_path = os.path.join(root_path, filename)

    return full_path, filename
