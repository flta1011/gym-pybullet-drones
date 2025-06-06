# -*- coding: utf-8 -*-
#
# ,---------,       ____  _ __
# |  ,-^-,  |      / __ )(_) /_______________ _____  ___
# | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
# | / ,--'  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
# Copyright (C) 2024 Bitcraze AB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, in version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
import unittest
from unittest.mock import ANY
from unittest.mock import mock_open
from unittest.mock import patch

import yaml

from cflib.localization import ParamFileManager


class TestParamFileManager(unittest.TestCase):
    def setUp(self):
        self.data = {
            "type": "persistent_param_state",
            "version": "1",
        }

    @patch("yaml.safe_load")
    def test_that_read_open_correct_file(self, mock_yaml_load):
        # Fixture
        mock_yaml_load.return_value = self.data
        file_name = "some/name.yaml"

        # Test
        with patch("builtins.open", new_callable=mock_open()) as mock_file:
            ParamFileManager.read(file_name)

        # Assert
        mock_file.assert_called_with(file_name, "r")

    @patch("yaml.safe_load")
    def test_that_missing_file_type_raises(self, mock_yaml_load):
        # Fixture
        self.data.pop("type")
        mock_yaml_load.return_value = self.data

        # Test
        # Assert
        with self.assertRaises(Exception):
            with patch("builtins.open", new_callable=mock_open()):
                ParamFileManager.read("some/name.yaml")

    @patch("yaml.safe_load")
    def test_that_wrong_file_type_raises(self, mock_yaml_load):
        # Fixture
        self.data["type"] = "wrong_type"
        mock_yaml_load.return_value = self.data

        # Test
        # Assert
        with self.assertRaises(Exception):
            with patch("builtins.open", new_callable=mock_open()):
                ParamFileManager.read("some/name.yaml")

    @patch("yaml.safe_load")
    def test_that_missing_version_raises(self, mock_yaml_load):

        # Fixture
        self.data.pop("version")
        mock_yaml_load.return_value = self.data

        # Test
        # Assert
        with self.assertRaises(Exception):
            with patch("builtins.open", new_callable=mock_open()):
                ParamFileManager.read("some/name.yaml")

    @patch("yaml.safe_load")
    def test_that_wrong_version_raises(self, mock_yaml_load):
        # Fixture
        self.data["version"] = "wrong_version"
        mock_yaml_load.return_value = self.data

        # Test
        # Assert
        with self.assertRaises(Exception):
            with patch("builtins.open", new_callable=mock_open()):
                ParamFileManager.read("some/name.yaml")

    @patch("yaml.safe_load")
    def test_that_no_data_returns_empty_default_data(self, mock_yaml_load):
        # Fixture
        mock_yaml_load.return_value = self.data

        # Test
        with patch("builtins.open", new_callable=mock_open()):
            actual_params = ParamFileManager.read("some/name.yaml")

        # Assert
        self.assertEqual(0, len(actual_params))

    @patch("yaml.dump")
    def test_file_end_to_end_write_read(self, mock_yaml_dump):
        # Fixture
        fixture_file = "test/localization/fixtures/parameters.yaml"

        file = open(fixture_file, "r")
        expected = yaml.safe_load(file)
        file.close()

        # Test
        params = ParamFileManager.read(fixture_file)
        with patch("builtins.open", new_callable=mock_open()):
            ParamFileManager.write("some/name.yaml", params=params)

            # Assert
            mock_yaml_dump.assert_called_with(expected, ANY)

    @patch("yaml.dump")
    def test_file_write_to_correct_file(self, mock_yaml_dump):
        # Fixture
        file_name = "some/name.yaml"

        # Test
        with patch("builtins.open", new_callable=mock_open()) as mock_file:
            ParamFileManager.write(file_name)

            # Assert
            mock_file.assert_called_with(file_name, "w")
