"""
당구 타점 추천 이미지 PNG 생성기
pip install pillow numpy 필요
"""
from PIL import Image, ImageDraw, ImageFont
import math

def create_billiard_hit_point_image():
    # 이미지 크기 - 고해상도 유지하면서 DPI 높임
    width = 2400  # 2배 해상도
    height = 1200  # 2배 해상도
    
    # 이미지 생성 (DPI 300으로 설정하면 Word에서 선명하게 출력)
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # 배경 그라디언트 효과 (녹색)
    for y in range(height):
        color_value = int(15 + (26 - 15) * (y / height))
        draw.rectangle([(0, y), (width, y+1)], fill=(color_value, 122, 74))
    
    # 왼쪽: 당구대 (크기 2배)
    table_x, table_y = 100, 200
    table_w, table_h = 1000, 800
    
    # 당구대 배경
    draw.rectangle([(table_x, table_y), (table_x + table_w, table_y + table_h)], 
                   fill=(15, 122, 74))
    
    # 테두리 (두께 증가)
    for i in range(10):
        draw.rectangle([(table_x - i, table_y - i), 
                       (table_x + table_w + i, table_y + table_h + i)], 
                       outline=(139, 69, 19), width=3)
    
    # 예상 경로 (노란색 점선)
    path_points = [
        (table_x + 90, table_y + 280),
        (table_x + 260, table_y + 60),
        (table_x + 380, table_y + 180)
    ]
    
    for i in range(len(path_points) - 1):
        x1, y1 = path_points[i]
        x2, y2 = path_points[i + 1]
        
        # 점선 그리기
        steps = 20
        for j in range(0, steps, 2):
            t1 = j / steps
            t2 = min((j + 1) / steps, 1)
            sx = x1 + (x2 - x1) * t1
            sy = y1 + (y2 - y1) * t1
            ex = x1 + (x2 - x1) * t2
            ey = y1 + (y2 - y1) * t2
            draw.line([(sx, sy), (ex, ey)], fill=(251, 191, 36), width=8)
    
    # 공 그리기 함수 (크기 2배)
    def draw_ball(x, y, radius, color, light_color, label=None):
        # 그림자
        draw.ellipse([(x - radius + 4, y + radius - 6), 
                     (x + radius + 4, y + radius + 6)], 
                     fill=(0, 0, 0, 50))
        
        # 공 본체
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], 
                    fill=color, outline=(200, 200, 200), width=4)
        
        # 하이라이트
        hl_r = radius // 3
        draw.ellipse([(x - radius//2 - hl_r, y - radius//2 - hl_r),
                     (x - radius//2 + hl_r, y - radius//2 + hl_r)],
                     fill=light_color)
        
        # 레이블
        if label:
            try:
                # 한글 폰트 시도
                try:
                    font = ImageFont.truetype("malgun.ttf", 32)
                except:
                    try:
                        font = ImageFont.truetype("gulim.ttf", 32)
                    except:
                        font = ImageFont.truetype("NanumGothic.ttf", 32)
            except:
                font = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text((x - text_w//2, y - text_h//2), label, 
                     fill='white', font=font)
    
    # 큐볼 (크기 2배)
    draw_ball(table_x + 180, table_y + 560, 44, (220, 220, 220), 
              (255, 255, 255), 'Cue')
    
    # 방해공
    draw_ball(table_x + 400, table_y + 400, 44, (148, 163, 184), 
              (203, 213, 225))
    
    # 목표공 (빨간색)
    draw_ball(table_x + 760, table_y + 360, 44, (239, 68, 68), 
              (252, 165, 165), '1')
    
    # 오른쪽: 3D 큐볼 + 타점 (크기 2배)
    ball_center_x = 1800
    ball_center_y = 600
    ball_radius = 260
    
    # 흰색 패널 배경
    panel_x = 1300
    draw.rectangle([(panel_x, 0), (width, height)], fill='white')
    
    # 제목들 (폰트 크기 2배)
    try:
        # Windows 한글 폰트 시도
        title_font = ImageFont.truetype("malgun.ttf", 40)  # 맑은 고딕
        subtitle_font = ImageFont.truetype("malgun.ttf", 28)
    except:
        try:
            title_font = ImageFont.truetype("gulim.ttf", 40)  # 굴림
            subtitle_font = ImageFont.truetype("gulim.ttf", 28)
        except:
            try:
                title_font = ImageFont.truetype("NanumGothic.ttf", 40)  # 나눔고딕
                subtitle_font = ImageFont.truetype("NanumGothic.ttf", 28)
            except:
                # 한글 폰트 없으면 영문으로 대체
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
                print("⚠️ 한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
    
    draw.text((160, 60), "Current Situation", fill='white', font=title_font)
    draw.text((1500, 60), "AI Recommended Hit Point", fill=(51, 51, 51), font=title_font)
    
    # 그림자 (크기 2배)
    shadow_offset = 30
    for i in range(shadow_offset):
        alpha = int(20 * (1 - i / shadow_offset))
        r = ball_radius * (0.9 - i * 0.01)
        draw.ellipse([
            (ball_center_x - r, ball_center_y + 280 + i),
            (ball_center_x + r, ball_center_y + 290 + i)
        ], fill=(200, 200, 200))
    
    # 3D 큐볼 그리기 (그라디언트 효과)
    for r in range(ball_radius, 0, -1):
        # 그라디언트 계산
        ratio = r / ball_radius
        brightness = int(180 + (255 - 180) * (1 - ratio) ** 0.5)
        color = (brightness, brightness, brightness)
        
        offset_x = ball_radius * 0.3 * (1 - ratio)
        offset_y = ball_radius * 0.3 * (1 - ratio)
        
        draw.ellipse([
            (ball_center_x - r + offset_x, ball_center_y - r + offset_y),
            (ball_center_x + r + offset_x, ball_center_y + r + offset_y)
        ], fill=color)
    
    # 외곽선 (두께 증가)
    draw.ellipse([
        (ball_center_x - ball_radius, ball_center_y - ball_radius),
        (ball_center_x + ball_radius, ball_center_y + ball_radius)
    ], outline=(150, 150, 150), width=6)
    
    # 타점 계산 (10시 방향 - 좌측 상단)
    # 시계 기준: 12시=위(270도), 3시=오른쪽(0도), 6시=아래(90도), 9시=왼쪽(180도)
    # 10시 방향 = 12시에서 왼쪽으로 60도 = 270도 + 60도 = 330도
    hit_angle = math.radians(210)  # 10시 방향 (좌측 상단)
    hit_distance = ball_radius * 0.65
    hit_x = ball_center_x + math.cos(hit_angle) * hit_distance
    hit_y = ball_center_y + math.sin(hit_angle) * hit_distance
    
    # 타점 표시 (빨간 점) - 크기 2배
    hit_radius = 48
    
    # 타점 그림자
    draw.ellipse([
        (hit_x - hit_radius + 4, hit_y - hit_radius + 4),
        (hit_x + hit_radius + 4, hit_y + hit_radius + 4)
    ], fill=(200, 200, 200))
    
    # 타점 본체
    draw.ellipse([
        (hit_x - hit_radius, hit_y - hit_radius),
        (hit_x + hit_radius, hit_y + hit_radius)
    ], fill=(220, 53, 69), outline='white', width=8)
    
    # 타점 중심점
    draw.ellipse([
        (hit_x - 8, hit_y - 8),
        (hit_x + 8, hit_y + 8)
    ], fill='white')
    
    # 화살표 (크기 2배)
    arrow_angle = math.atan2(ball_center_y - hit_y, ball_center_x - hit_x)
    arrow_start = 64
    arrow_length = 110
    
    arrow_start_x = hit_x + math.cos(arrow_angle) * arrow_start
    arrow_start_y = hit_y + math.sin(arrow_angle) * arrow_start
    arrow_end_x = arrow_start_x + math.cos(arrow_angle) * arrow_length
    arrow_end_y = arrow_start_y + math.sin(arrow_angle) * arrow_length
    
    # 화살표 선 (두께 증가)
    draw.line([
        (arrow_start_x, arrow_start_y),
        (arrow_end_x, arrow_end_y)
    ], fill=(220, 53, 69), width=12)
    
    # 화살표 머리 (크기 2배)
    head_length = 32
    head_angle = math.pi / 6
    points = [
        (arrow_end_x, arrow_end_y),
        (arrow_end_x - head_length * math.cos(arrow_angle - head_angle),
         arrow_end_y - head_length * math.sin(arrow_angle - head_angle)),
        (arrow_end_x - head_length * math.cos(arrow_angle + head_angle),
         arrow_end_y - head_length * math.sin(arrow_angle + head_angle))
    ]
    draw.polygon(points, fill=(220, 53, 69))
    
    # 정보 패널 (크기/위치 2배)
    info_y = 1000
    draw.rectangle([(panel_x + 60, info_y), (width - 60, info_y + 160)],
                   fill=(248, 249, 250), outline=(15, 122, 74), width=6)
    
    # 텍스트 정보
    draw.text((panel_x + 50, info_y + 10), "AI Optimal Hit Point Analysis", 
             fill=(102, 102, 102), font=subtitle_font)
    draw.text((panel_x + 50, info_y + 35), "Left-Top (10 o'clock)", 
             fill=(15, 122, 74), font=title_font)
    
    # 통계
    stats = [
        ("Power", "68%"),
        ("Angle", "35°"),  # degree 기호는 유지 가능
        ("Success", "76%")
    ]
    
    stat_x = panel_x + 70
    for i, (label, value) in enumerate(stats):
        x = stat_x + i * 130
        draw.text((x, info_y + 60), f"{label}: {value}", 
                 fill=(102, 102, 102), font=subtitle_font)
    
    return img

# 이미지 생성 및 저장
if __name__ == "__main__":
    print("이미지 생성 중...")
    img = create_billiard_hit_point_image()
    
    # PNG로 저장 (DPI 300으로 고해상도)
    filename = "billiard_ai_hit_point.png"
    img.save(filename, "PNG", dpi=(300, 300))
    print(f"✅ 고해상도 이미지가 생성되었습니다: {filename}")
    print(f"크기: 2400x1200 픽셀 (300 DPI)")
    print(f"Word에서 작은 크기로 삽입해도 선명합니다!")
