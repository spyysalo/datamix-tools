# datamix-tools
Tools for creating and managing data mixtures for model training

1. Basic Usage (Print to Console)

To calculate and print the data path in terminal:

```bash
python3 datamix.py <mix_config> <path_mapping>
```
Example:

```bash
python3 datamix.py mixes/1TT-option-1.json paths/lumi-gpt-oss-paths.json
```
2. Save to File

To save the generated string for direct import into a Megatron-LM training script or environment variable:

```bash
python3 datamix.py mixes/1TT-option-1.json paths/lumi-gpt-oss-paths.json --output datapath.txt
```