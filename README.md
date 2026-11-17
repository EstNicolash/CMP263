
# CMP263

## Como Reproduzir

Este projeto utiliza **Nix Flakes** (`flake.nix` e `flake.lock`) para garantir um ambiente determinístico e reproduzível, tanto para desenvolvimento interativo quanto para a geração de uma imagem de container. Também é fornecido um arquivo `requirements.txt` para execução em ambientes Python tradicionais.

---
### 1. Pré-requisitos

####  Instalação do Nix
Caso ainda não possua o Nix instalado, siga as instruções da [página oficial de download do Nix](https://nixos.org/download/):

- **Linux / macOS / WSL2 (instalação multiusuário recomendada):**
```bash
  curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install | sh -s -- --daemon
```

Habilite o suporte a Flakes adicionando a linha abaixo ao arquivo `~/.config/nix/nix.conf` ou `/etc/nix/nix.conf` (caso ainda não esteja ativo):
    
   
```bash
    experimental-features = nix-command flakes
```
#### Clonar o Repositório

```bash
git clone https://github.com/EstNicolash/CMP263.git
cd CMP263
```

---

### 2. Opção A — Execução Automatizada via Container (Podman / Docker) *(Recomendado)*

Esta abordagem reproduz todo o pipeline de ponta a ponta (`main.py`) em um container.

#### Pré-requisitos

* **Podman** (ou **Docker**) instalado.

#### Construir e Executar

1. Gere a imagem OCI a partir do `flake.lock`:
```bash
nix build .#container

```


2. Carregue a imagem no Podman:
```bash
podman load < result

```


3. Execute o pipeline montando o diretório atual para salvar os artefatos localmente:
```bash
podman run --rm -v "$(pwd):/app:Z" heart-disease-ml:latest

```


*(Caso utilize Docker no lugar do Podman, substitua `podman` por `docker` e remova o sufixo `:Z` do volume).*

---

### 3. Opção B — Ambiente Local com Nix (`main.py` ou Jupyter Lab)

1. Carregue o ambiente de desenvolvimento:
```bash
nix develop

```


*(Se utilizar `direnv`, basta rodar `direnv allow` na raiz do projeto).*

2. Execute o script completo pelo terminal:
```bash
python main.py

```


3. **Ou**, para explorar o notebook interativamente:
```bash
jupyter lab

```


* Abra `notebooks/Predição_de_Doença_Cardíaca_com_Aprendizagem_de_Máquina.ipynb`, selecione o kernel Python do Nix e execute as células.



---

### 4. Opção C — Ambiente Python Tradicional (`venv` + `pip`)

Caso não possua Nix ou Podman instalados:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py

```

---

### Estrutura de Artefatos Gerados

A execução do `main.py` ou do notebook salva automaticamente os resultados nas seguintes pastas:

* `models/`: Modelo serializado final (`melhor_modelo_heart_disease.pkl`) e hiperparâmetros ótimos (`params/*.json`).
* `reports/metrics/`: Tabelas comparativas de modelos e métricas finais (`.csv`).
* `reports/figures/`: Gráficos de interpretabilidade SHAP e importância por permutação (`.png`).
