call mkvirtualenv testenv
call workon testenv
call pip install laura pytest
call pytest -x laura\tests\tests.py
call deactivate
call rmdir %WORKON_HOME%\testenv /s /q