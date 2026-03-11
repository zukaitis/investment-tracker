![Static Badge](https://img.shields.io/badge/version-1.0.0-gree?link=https%3A%2F%2Fgaidosas.org)

# Generate a daily report about your financial portfolio by using Github Actions

Project uses [yfinance](https://github.com/ranaroussi/yfinance) to acquire market data and [plotly](https://plotly.com/python/) to display graphs. Daily example reports can be found under [Releases](https://github.com/zukaitis/investment-tracker/releases) 

## Usage

1. [Create a new **private** repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository) on your Github account
2. Create a `.github/workflows/generate_report.yaml` file in your repository and copy all the content of [file from this repository](https://github.com/zukaitis/investment-tracker/blob/release/.github/workflows/generate_report.yaml) into it
3. Add files containing asset information to the root of your repository. Examples of the input format can be found in [`examples`](https://github.com/zukaitis/investment-tracker/tree/release/examples) directory
4. Create a `settings.yaml` file in the root of your repository and adjust the report settings to your liking. Example of settings file can be found [here](https://github.com/zukaitis/investment-tracker/blob/release/input_data/EXAMPLE_settings.yaml)
5. Adjust report generation settings (e.g. daily generation time) in your own `.github/workflows/generate_report.yaml` file
6. Star this repository, so I would know people are using the project
