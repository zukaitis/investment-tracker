[![Static Badge](https://img.shields.io/badge/version-1.2.0-green)](https://github.com/zukaitis/investment-tracker/releases/tag/1.2.0)

# Generate a daily report about your financial portfolio by using Github Actions

Project uses [yfinance](https://github.com/ranaroussi/yfinance) to acquire market data and [Plotly](https://plotly.com/python/) to display graphs. Daily example reports can be found under [Releases](https://github.com/zukaitis/investment-tracker/releases) 

## Usage

1. [Create a new **private** repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository) on your Github account
2. Create a `.github/workflows/generate_report.yaml` file in your repository and copy all the content of [file from this repository](https://github.com/zukaitis/investment-tracker/blob/release/.github/workflows/generate_report.yaml) into it
3. Add files containing information about your assets to the root of your repository - one file per asset. Examples of the input format can be found in [`examples`](https://github.com/zukaitis/investment-tracker/tree/release/examples) directory. Fields are explained in the [`Input_Fields_Explained.yaml`](https://github.com/zukaitis/investment-tracker/blob/release/examples/Input_Fields_Explained.yaml) example
4. Create a `settings.yaml` file in the root of your repository and adjust the report settings to your liking. Example of settings file can be found [here](https://github.com/zukaitis/investment-tracker/blob/release/examples/settings.yaml)
5. (Optional) Adjust report generation settings (e.g. daily generation time) in your own `.github/workflows/generate_report.yaml` file
6. Star this repository, so I would know people are using the project
