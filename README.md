```bash
docker build -t data-science-container .
```
```bash
docker run --gpus all -p 8811:8811 -v "$(pwd):/app" --name data-science-container -d data-science-container:latest   
```