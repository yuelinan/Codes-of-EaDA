Official implementation of "Empowering Federated Graph Rationale Learning withLatent Environments".

## Data download
- Spurious-Motif: this dataset can be generated via `spmotif_gen/spmotif.ipynb` in [DIR](https://github.com/Wuyxin/DIR-GNN/tree/main). 
- Open Graph Benchmark (OGB): this dataset can be downloaded when running run.sh.


## How to run EaDA?

To train EaDA on the OGB datasets:

```python
sh run.sh
```
To train FedGR on Spurious-Motif dataset:

```python
# cd spmotif_codes
sh run.sh
```





