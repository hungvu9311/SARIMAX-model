RUN
create venv: python3 -m venv venv
active: 
    - window powershell: .\venv\Scripts\Activate.ps1
    - ubuntu: source ....
install: pip install -r .\requirements.txt

