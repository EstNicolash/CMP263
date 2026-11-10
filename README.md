# CMP263

## Como Reproduzir 

Este projeto utiliza o gerenciador de pacotes **Nix** com **Flakes** para garantir um ambiente determinístico e reproduzível (incluindo Python, bibliotecas de Machine Learning, Jupyter e dependências de sistema).

### 1. Pré-requisitos (Instalação do Nix)

Caso ainda não possua o Nix instalado no seu sistema:

- **Linux / macOS:**
  ```bash
  curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install | sh -s -- --daemon
  ```

- Habilite o suporte a Flakes adicionando ao arquivo `~/.config/nix/nix.conf` ou `/etc/nix/nix.conf` (se já não estiver ativo):
  ```conf
  experimental-features = nix-command flakes
  ```

---

### 2. Entrar no Ambiente de Desenvolvimento

Clone o repositório e acesse o diretório do projeto:

```bash
git clone https://github.com/EstNicolash/CMP263.git
cd CMP263
```

Carregue as dependências do ambiente:

- **Sem direnv (padrão):**
  ```bash
  nix develop
  ```

- **Com direnv (automático):**
  Se você utiliza `direnv`, basta autorizar a leitura do ambiente local:
  ```bash
  direnv allow
  ```

---

### 3. Executar o Jupyter Lab

Com o ambiente do Nix carregado no terminal, inicie o Jupyter Lab a partir da raiz:

```bash
jupyter lab
```

1. Na barra lateral do navegador, abra a pasta `notebooks/`.
2. Acesse o arquivo `Predição_de_Doença_Cardíaca_com_Aprendizagem_de_Máquina.ipynb`.
3. Selecione o kernel padrão do Python fornecido pelo Nix e execute as células.

---

### Estrutura de Artefatos Gerados

As execuções do notebook salvam os resultados automaticamente nas seguintes pastas:

- `models/`: Modelo serializado final (`.pkl`) e hiperparâmetros ótimos (`params/*.json`).
- `reports/metrics/`: Tabelas comparativas de modelos e métricas finais (`.csv`).
- `reports/figures/`: Gráficos gerados.
