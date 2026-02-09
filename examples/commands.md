# Example Commands

```bash
# generate sample data
tpuopt sample-data

# analyze
 tpuopt analyze \
  --profile_dir ./sample_data \
  --model_name demo-model \
  --workload inference \
  --out_dir ./outputs

# report
 tpuopt report --input ./outputs/summary.json --out_dir ./report
```

Expected outputs:
- `./outputs/summary.json`
- `./outputs/recommendations.md`
- `./outputs/charts/step_time.html`
- `./report/report.md`
