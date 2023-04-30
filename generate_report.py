#!/usr/bin/python3

import argparse
import enum
import os
import yaml

from source.settings import Settings
from source import report
from source import dataset


class Main:
    def __init__(self):
        self.settings = Settings()
        self.dataset = dataset.Dataset(self.settings)

    def run(self):
        # pd.set_option('display.max_rows', None)  # makes pandas print all dataframe rows

        self._parse_arguments()
        self._read_settings()
        self._read_asset_data()

        print(self.dataset.assets)
        self.dataset.get_historical_data_sum(self.dataset.assets).to_csv("out.csv")

    def _parse_arguments(self):
        default_input_dir = (
            f"{os.path.dirname(os.path.realpath(__file__))}{os.path.sep}less_data"
        )

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input_dir",
            "-i",
            type=str,
            default=default_input_dir,
            help="Directory, containing input files",
        )
        for s in self.settings:  # all settings are possible arguments
            parser.add_argument(
                f"--{s}",
                type=type(getattr(self.settings, s)),
                help=self.settings.get_description(s),
                choices=self.settings.get_allowed(s),
            )
        self.arguments = parser.parse_args()
        for s in self.settings:
            if getattr(self.arguments, s) != None:
                setattr(self.settings, s, getattr(self.arguments, s))

    def _read_settings(self):
        class SettingsFound(enum.Enum):
            NO = enum.auto()
            YES = enum.auto()
            WARNED = enum.auto()

        settings_found = SettingsFound.NO
        for entry in os.scandir(self.arguments.input_dir):
            if entry.is_file() and entry.path.endswith((".yaml", ".yml")):
                with open(entry, "r") as entry_file:
                    input = yaml.safe_load(entry_file)
                # check if there are general settings in file
                if ("data" not in input) and (any([s in input for s in self.settings])):
                    report.report(f"Reading setting file {entry.name}")
                    for s in input:
                        setattr(self.settings, s, input[s])
                    if settings_found == SettingsFound.YES:
                        report.warn("Multiple settings files detected, expect trouble")
                        settings_found = (
                            SettingsFound.WARNED
                        )  # three states, so error would only pop once
                    elif settings_found == SettingsFound.NO:
                        settings_found = SettingsFound.YES

    def _read_asset_data(self):
        for entry in os.scandir(self.arguments.input_dir):
            if entry.is_file() and entry.path.endswith((".yaml", ".yml")):
                with open(entry, "r") as entry_file:
                    datadict = yaml.safe_load(entry_file)
                # check that there are no general settings in file
                if "data" in datadict:
                    report.report(f"Reading asset data file {entry.name}")
                    datadict["filename"] = entry.name
                    if ("name" not in datadict) or (
                        not isinstance(datadict["name"], str)
                    ):
                        datadict["name"] = os.path.splitext(entry.name)[
                            0
                        ]  # remove extension
                    try:
                        self.dataset.append(datadict)
                    except ValueError as error:
                        report.error(error)


if __name__ == "__main__":
    main = Main()
    main.run()
