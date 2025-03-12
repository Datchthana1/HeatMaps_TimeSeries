import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv(
    rf"D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\partition_combined_data_upsampled_pm_3H_spline_1degreeM.csv"
)  # ลดความซับซ้อนของชื่อไฟล์

# แปลง DateTime ให้เป็นรูปแบบวันที่ที่ถูกต้อง
df["DateTime"] = pd.to_datetime(
    df["DateTime"], dayfirst=True
)  # สร้างคอลัมน์วันที่ในรูปแบบที่ pandas เข้าใจ

# สร้างคอลัมน์เพิ่มเติมสำหรับการวิเคราะห์
df["Year"] = df["DateTime"].dt.year  # แยกปีจาก DateTime
df["Month"] = df["DateTime"].dt.month  # แยกเดือนจาก DateTime
df["Hour"] = df["DateTime"].dt.hour  # แยกชั่วโมงจาก DateTime
df["MonthYear"] = df["DateTime"].dt.strftime("%Y-%m")  # สร้างคอลัมน์ในรูปแบบ ปี-เดือน
df["DayOfWeek"] = df["DateTime"].dt.dayofweek  # สร้างคอลัมน์วันในสัปดาห์ (0=จันทร์, 6=อาทิตย์)
print(f"Dataframe: \n{df}")

# ตั้งค่าชื่อเดือนและวันสำหรับใช้ในกราฟ
month_names = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ---------- 1. สร้าง Heatmap แสดงค่าเฉลี่ย PM2.5 รายเดือนแยกตามปี ----------
# สร้าง pivot table โดยให้แกน x เป็นเดือน แกน y เป็นปี และค่าคือค่าเฉลี่ย PM2.5
yearly_monthly = df.pivot_table(
    index="Year", columns="Month", values="PM2.5", aggfunc="mean"
)

# สร้างกราฟขนาด 14x6 นิ้ว
plt.figure(figsize=(14, 6))

# วาด heatmap โดยใช้ colormap แบบ YlOrRd (เหลือง-ส้ม-แดง) ซึ่งเหมาะกับการแสดงค่ามลพิษ
# annot=True หมายถึงแสดงค่าตัวเลขบนแต่ละเซลล์
# fmt=".1f" คือแสดงตัวเลขทศนิยม 1 ตำแหน่ง
ax = sns.heatmap(
    yearly_monthly,
    cmap="YlOrRd",
    annot=True,  # การตั้งค่าค่านี้เป็น True จะทำให้แสดงค่าตัวเลขในแต่ละเซลล์ของ Heatmap
    fmt=".2f",  # พารามิเตอร์นี้ใช้เพื่อกำหนดสีของ Heatmap
    linewidths=0.5,  # กำหนดความหนาของเส้นขอบที่แบ่งแต่ละเซลล์ใน Heatmap
    cbar_kws={"label": "PM2.5 (µg/m³)"},  # กำหนดป้ายกำกับแถบสี
)

# กำหนดชื่อกราฟและแกน
plt.title("Monthly Average PM2.5 Values (2016-2024)", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Year", fontsize=12)

# เปลี่ยนชื่อเดือนจากตัวเลขเป็นชื่อย่อ
ax.set_xticklabels(month_names)

# จัดรูปแบบกราฟให้สวยงาม
plt.tight_layout()
# บันทึกกราฟเป็นไฟล์รูปภาพ
plt.savefig("pm25_yearly_monthly_heatmap.png", dpi=300)
plt.show()  # แสดงกราฟ

# ---------- 2. สร้าง Heatmap แสดงค่าเฉลี่ย PM2.5 ตามชั่วโมงและวันในสัปดาห์ ----------
# สร้าง pivot table โดยให้แกน x เป็นวันในสัปดาห์ แกน y เป็นชั่วโมง และค่าคือค่าเฉลี่ย PM2.5
hourly_dow = df.pivot_table(
    index="Hour", columns="DayOfWeek", values="PM2.5", aggfunc="mean"
)

plt.figure(figsize=(12, 8))
ax = sns.heatmap(
    hourly_dow,
    cmap="YlOrRd",
    annot=True,
    fmt=".1f",
    linewidths=0.5,
    cbar_kws={"label": "PM2.5 (µg/m³)"},
)

plt.title("Average PM2.5 by Hour of Day and Day of Week", fontsize=14)
plt.xlabel("Day of Week", fontsize=12)
plt.ylabel("Hour of Day", fontsize=12)

# เปลี่ยนป้ายกำกับจากตัวเลขเป็นชื่อวัน
ax.set_xticklabels(days)

plt.tight_layout()
plt.savefig("pm25_hourly_weekly_heatmap.png", dpi=300)
plt.show()

# ---------- 3. สร้าง Heatmap แสดงความสัมพันธ์ระหว่างความเร็วลมและความชื้นต่อ PM2.5 ----------
# แบ่งช่วง PM2.5 เป็นกลุ่มตามระดับคุณภาพอากาศ
bins = [0, 25, 50, 100, np.inf]  # กำหนดช่วงของ PM2.5
labels = [
    "Good (0-25)",
    "Moderate (25-50)",
    "Unhealthy (50-100)",
    "Hazardous (>100)",
]  # กำหนดชื่อช่วง
df["PM_Category"] = pd.cut(
    df["PM2.5"], bins=bins, labels=labels
)  # สร้างคอลัมน์หมวดหมู่ PM2.5

# สร้าง pivot table แสดงความสัมพันธ์ระหว่างความเร็วลมและความชื้น
# แบ่งความเร็วลมและความชื้นเป็นช่วงๆ โดยใช้ pd.cut
ws_hm = df.pivot_table(
    index=pd.cut(df["WS"], bins=[0, 2, 4, 6, 8, np.inf]),  # แบ่งความเร็วลมเป็น 5 ช่วง
    columns=pd.cut(df["HM"], bins=[0, 25, 50, 75, 100]),  # แบ่งความชื้นเป็น 4 ช่วง
    values="PM2.5",  # ใช้ค่า PM2.5 เป็นค่าที่แสดงในแต่ละเซลล์
    aggfunc="mean",  # คำนวณค่าเฉลี่ยของแต่ละกลุ่ม
)

plt.figure(figsize=(14, 6))
sns.heatmap(
    ws_hm,
    cmap="YlOrRd",  # กำหนด color map
    annot=True,  # แสดงค่าตัวเลข
    fmt=".1f",  # รูปแบบตัวเลข (ทศนิยม 1 ตำแหน่ง)
    linewidths=0.5,  # ความหนาของเส้นแบ่งเซลล์
    cbar_kws={"label": "PM2.5 (µg/m³)"},  # กำหนดป้ายกำกับแถบสี
)

plt.title("Average PM2.5 by Wind Speed and Humidity", fontsize=14)
plt.xlabel("Humidity (%)", fontsize=12)
plt.ylabel("Wind Speed (m/s)", fontsize=12)

plt.tight_layout()
plt.savefig("pm25_ws_hm_heatmap.png", dpi=300)
plt.show()
