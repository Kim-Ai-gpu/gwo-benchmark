import pandas as pd
from jinja2 import Environment, FileSystemLoader
import os

# HTML 템플릿 파일명 및 출력 파일명 설정
TEMPLATE_FILE = "leaderboard_template.html"
OUTPUT_FILE = "index.html"
CSV_FILE = "results.csv"

def get_tier_badge(tier):
    color_map = {
        'S': 'background: linear-gradient(to right, #f83600 0%, #f9d423 100%); color: white;',
        'A': 'background-color: #4CAF50; color: white;',
        'B': 'background-color: #2196F3; color: white;',
        'C': 'background-color: #ff9800; color: white;',
        'D': 'background-color: #607D8B; color: white;',
    }
    style = color_map.get(tier, 'background-color: #f44336; color: white;')
    return f'<span class="tier-badge" style="{style}">{tier}</span>'

def main():
    if not os.path.exists(CSV_FILE):
        print(f"Error: '{CSV_FILE}' not found!")
        return
        
    df = pd.read_csv(CSV_FILE)

    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    
    records = df.to_dict(orient='records')
    
    for record in records:
        record['tier_badge'] = get_tier_badge(record['tier'])
        record['score'] = f"{record['score']:.2f}"
        record['test_acc'] = f"{record['test_acc']:.2f}%"
        record['omega_proxy'] = f"{record['omega_proxy']:.2f}"
        record['latency_ms'] = f"{record['latency_ms']:.2f}"
        record['c_p_m'] = f"{record['c_p_m']:.3f}"


    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(TEMPLATE_FILE)
    
    html_content = template.render(
        records=records,
        page_title="GWO Benchmark: The Architect's Arena"
    )
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Successfully generated '{OUTPUT_FILE}'!")


if __name__ == "__main__":
    main()