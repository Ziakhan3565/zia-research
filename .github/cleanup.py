from pathlib import Path
p = Path('dashboard.py')
s = p.read_text(encoding='utf-8')
block = 'AUTO_SCAN_TFS = ("15M", "1H", "4H")\nAUTO_SCAN_INTERVAL_SECONDS = 10\n'
first = s.find(block)
if first < 0:
    raise SystemExit('auto scan constants not found')
rest = s[first + len(block):].replace(block, '')
s = s[:first] + block + rest
p.write_text(s, encoding='utf-8')
