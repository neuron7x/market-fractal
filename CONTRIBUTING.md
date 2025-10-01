# Contributing Guidelines

Дякуємо, що хочете зробити внесок у **BTC Market Intelligence Fractal** 🚀

## Як працювати з кодом
- Використовуйте Python ≥ 3.11.
- Усі залежності мають бути встановлені через:
  ```bash
  pip install -r requirements.txt -c constraints.txt
  pip install .[dev] -c constraints.txt
  ```
- Перед комітом запускайте перевірки стилю та форматування:
  ```bash
  black .
  ruff check .
  ```
- Перевіряйте функціональність тестами:
  - швидка димова перевірка: `pytest -m smoke`
  - повна регресія перед релізом/PR: `pytest`

## Оновлення залежностей
- У `pyproject.toml` та `requirements.txt` вказуйте мінімальні підтримувані версії (`>=` або `~=`).
- Точні версії фіксуйте лише у `constraints.txt` для відтворюваності CI.
- За потреби перегенеруйте `requirements.txt` на основі `pyproject.toml`:
  ```bash
  pip-compile pyproject.toml --output-file requirements.txt
  ```

