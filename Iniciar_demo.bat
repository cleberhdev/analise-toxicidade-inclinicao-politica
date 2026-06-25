@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=py
where py >nul 2>nul || set PY=python
if not exist ".venv\Scripts\python.exe" (
  echo Criando ambiente e instalando bibliotecas ^(apenas na 1a vez^)...
  %PY% -m venv .venv
  .venv\Scripts\python -m pip install -r requirements_api.txt
)
echo Iniciando o servidor de modelos...
start "Servidor de Modelos (NAO FECHE)" .venv\Scripts\python app.py
timeout /t 8 /nobreak >nul
start "" chrome "%~dp0site_projeto.html"
echo.
echo Pronto! Na pagina, role ate "Analisador". Deve mostrar "Modelo real conectado".
echo Para encerrar, feche a janela do Servidor de Modelos.
