# Tools

## Plot

The script allows you to plot data from multiple CSV files, creating subplots for columns matching specified patterns.


### Features

- **Multi-file Comparison:** Plot data from multiple CSV files side-by-side, organized by specified patterns.
- **Column Pattern Matching:** Match columns by name patterns with wildcards (e.g., `action_1`, `action_2`, etc.).
- **Custom Frequency Support:** Option to set the sampling frequency for each file.

### Usage

#### Running the Script
The basic format for running the script is:

```sh
python3 tools/plot.py --window [csv_file_1:pattern_1[:freq_1]::csv_file_2:pattern_2[:freq_2]::...]
```

Each `--window` argument specifies a comparison window between two or more files. You can use multiple `--window` arguments for separate figures.

- `csv_file`: The path to a CSV file.
- `pattern`: The column name prefix (e.g., `action_`, `speed_`).
- `freq` (optional): The frequency in Hz (default is 60 Hz)

#### Examples
**Example 1: Basic Comparison of Columns with Matching Patterns**

```sh
python3 tools/plot.py --window [../input_files/complete_data_1.csv:action_::../input_files/complete_data_2.csv:action_]
```

This command will plot columns starting with `action_` (e.g., `action_1`, `action_2`) from both CSV files side-by-side in subplots.

**Example 2: Multiple Patterns and Custom Frequencies**

```sh
python3 tools/plot.py \
  --window [../input_files/complete_data_1.csv:action_:50::../input_files/complete_data_2.csv:action_:60] \
  --window [../input_files/complete_data_1.csv:speed_:50::../input_files/complete_data_2.csv:speed_:60]
```

This command will:
- Create one figure comparing `action_` columns with frequencies 50 Hz and 60 Hz from each file.
- Create a second figure comparing `speed_` columns at 60 Hz from both files.

### Troubleshooting
- **File Not Found:** If the script cannot find a specified file, it will skip that window and proceed.
- **No Matching Columns:** If no columns match the specified pattern, the script will display a warning and skip that window.
- **Mismatch in Column Count:** If files have different numbers of matching columns, only the minimum number of matching columns will be plotted, and a warning will appear.

