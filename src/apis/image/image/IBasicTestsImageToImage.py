import abc


class IBasicTestsImageToImage(metaclass=abc.ABCMeta):
    """
    Class to test the background removal endpoint
    """

    target_url = None

    @abc.abstractmethod
    def test_image_input_task(self, model: str) -> bool:
        """
        Test the background removal endpoint with a jpg image input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        raise NotImplementedError

    @abc.abstractmethod
    def test_image_url_input_task(self, model: str) -> bool:
        """
        Test the background removal endpoint with a jpg image input retrieved from an url

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        raise NotImplementedError

    @abc.abstractmethod
    def test_invalid_image_input_task(self, model: str) -> bool:
        """
        Test the background removal endpoint with an invalid image input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        raise NotImplementedError

    @abc.abstractmethod
    def test_invalid_image_url_input_task(self, model: str) -> bool:
        """
        Test the background removal endpoint with an invalid image url input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        raise NotImplementedError

    @abc.abstractmethod
    def test_empty_input_task(self, model: str) -> bool:
        """
        Test the background removal endpoint with an empty input

        Args:
            model (str): model to test

        Returns:
            bool: True if the test passed, False otherwise
        """

        raise NotImplementedError
