set TELEGRAM_BOT_TOKEN=8445645385:AAGI-5NjeZhVKnGvmX5exJpkcF0QV61ECTo
set FOOTBALL_API_KEY=7b7473ea52e1039f64ef655864f5689c

ejemplos:

python predict.py "EQUIPO_LOCAL" "EQUIPO_VISITANTE" --sede "SEDE" --fase "FASE" --fecha "AAAA-MM-DD" --dias-local 7 --dias-visitante 7


python predict.py "South Africa" "Canada" --sede "Los Angeles" --fase "Dieciseisavos" --fecha "2026-06-28" --dias-local 3 --dias-visitante 3
python predict.py "France" "Sweden" --sede "Nueva York" --fase "Dieciseisavos" --fecha "2026-06-30" --dias-local 7 --dias-visitante 7