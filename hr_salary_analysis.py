# HR Salary Analytics - Is the Company Paying People Fairly?
# Industry: IT and HR Analytics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.rcParams['figure.facecolor'] = '#f8f9fa'
sns.set_palette("husl")

# ─────────────────────────────────────────────
# 1. GENERATE DATASET
# ─────────────────────────────────────────────
n = 500

designations = ['Junior Developer', 'Senior Developer', 'Team Lead', 'Manager', 'Director']
education_levels = ['Bachelor', 'Master', 'PhD']
locations = ['Bangalore', 'Hyderabad', 'Mumbai', 'Chennai', 'Pune']

base_salary = {
    'Junior Developer': 400000,
    'Senior Developer': 800000,
    'Team Lead': 1200000,
    'Manager': 1600000,
    'Director': 2500000,
}

designation = np.random.choice(designations, n, p=[0.30, 0.30, 0.20, 0.15, 0.05])
experience = np.array([
    np.random.randint(0, 3) if d == 'Junior Developer' else
    np.random.randint(3, 8) if d == 'Senior Developer' else
    np.random.randint(6, 12) if d == 'Team Lead' else
    np.random.randint(10, 18) if d == 'Manager' else
    np.random.randint(15, 25)
    for d in designation
])
education = np.random.choice(education_levels, n, p=[0.55, 0.35, 0.10])
location = np.random.choice(locations, n)
performance_score = np.random.randint(1, 6, n)  # 1 to 5

location_multiplier = {'Bangalore': 1.15, 'Mumbai': 1.10, 'Hyderabad': 1.00, 'Chennai': 0.95, 'Pune': 0.98}
edu_multiplier = {'Bachelor': 1.00, 'Master': 1.08, 'PhD': 1.15}

salary = np.array([
    base_salary[d]
    * (1 + 0.05 * exp)
    * location_multiplier[loc]
    * edu_multiplier[edu]
    * (1 + 0.03 * perf)
    + np.random.normal(0, 30000)
    for d, exp, loc, edu, perf in zip(designation, experience, location, education, performance_score)
])
salary = np.clip(salary, 250000, 5000000).astype(int)

salary_increment = np.array([
    round(perf * 2.5 + np.random.normal(0, 1.5), 1)
    for perf in performance_score
])
salary_increment = np.clip(salary_increment, 0, 20)

employee_id = [f"EMP{str(i+1).zfill(4)}" for i in range(n)]

df = pd.DataFrame({
    'Employee_ID': employee_id,
    'Designation': designation,
    'Experience': experience,
    'Education': education,
    'Location': location,
    'Performance_Score': performance_score,
    'Salary': salary,
    'Salary_Increment': salary_increment
})

df.to_csv('hr_salary_data.csv', index=False)
print("[OK] Dataset generated and saved as hr_salary_data.csv")

# ─────────────────────────────────────────────
# 2. DATA CLEANING & VALIDATION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: DATA CLEANING & VALIDATION")
print("="*60)

print(f"\nShape: {df.shape}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDuplicates: {df.duplicated().sum()}")
print(f"\nData Types:\n{df.dtypes}")

# Validate ranges
assert df['Performance_Score'].between(1, 5).all(), "Performance score out of range"
assert df['Experience'].ge(0).all(), "Negative experience found"
assert df['Salary'].ge(0).all(), "Negative salary found"
assert df['Salary_Increment'].between(0, 20).all(), "Increment out of range"
print("\n[OK] All validation checks passed.")

print(f"\nDescriptive Statistics:\n{df[['Experience','Salary','Performance_Score','Salary_Increment']].describe().round(2)}")

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

# Salary by Designation
print("\nAverage Salary by Designation:")
role_salary = df.groupby('Designation')['Salary'].agg(['mean','median','std','count']).round(0)
print(role_salary)

# Salary by Location
print("\nAverage Salary by Location:")
loc_salary = df.groupby('Location')['Salary'].agg(['mean','median']).round(0)
print(loc_salary)

# Salary Growth with Experience
print("\nSalary Growth with Experience (binned):")
df['Exp_Bin'] = pd.cut(df['Experience'], bins=[0,3,6,10,15,25], labels=['0-3','3-6','6-10','10-15','15+'])
print(df.groupby('Exp_Bin')['Salary'].mean().round(0))

# Performance vs Increment
print("\nAvg Salary Increment by Performance Score:")
print(df.groupby('Performance_Score')['Salary_Increment'].mean().round(2))

# ─────────────────────────────────────────────
# 4. OUTLIER DETECTION (Compensation Anomalies)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: OUTLIER DETECTION — COMPENSATION ANOMALIES")
print("="*60)

df['Z_Score'] = df.groupby('Designation')['Salary'].transform(
    lambda x: stats.zscore(x, ddof=1)
)
outliers = df[df['Z_Score'].abs() > 2]
print(f"\nEmployees with salary deviating >2 std from role peers: {len(outliers)}")
print(outliers[['Employee_ID','Designation','Experience','Location','Salary','Z_Score']].head(10).to_string(index=False))

# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: VISUALISATIONS")
print("="*60)

fig = plt.figure(figsize=(18, 14))
fig.suptitle("HR Salary Analytics Dashboard — Is the Company Paying Fairly?",
             fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── VIZ 1: Salary Distribution by Designation (Box Plot)
ax1 = fig.add_subplot(gs[0, 0])
order = ['Junior Developer', 'Senior Developer', 'Team Lead', 'Manager', 'Director']
sns.boxplot(data=df, x='Designation', y='Salary', order=order, ax=ax1, palette='Set2')
ax1.set_title("Viz 1: Salary Distribution by Job Role", fontweight='bold')
ax1.set_xlabel("Designation")
ax1.set_ylabel("Salary (₹)")
ax1.tick_params(axis='x', rotation=20)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x/100000:.0f}L'))

# ── VIZ 2: Salary Growth with Experience (Scatter + Trend)
ax2 = fig.add_subplot(gs[0, 1])
colors = {'Junior Developer': '#e74c3c', 'Senior Developer': '#3498db',
          'Team Lead': '#2ecc71', 'Manager': '#f39c12', 'Director': '#9b59b6'}
for role in order:
    subset = df[df['Designation'] == role]
    ax2.scatter(subset['Experience'], subset['Salary']/100000,
                label=role, alpha=0.5, s=25, color=colors[role])
z = np.polyfit(df['Experience'], df['Salary']/100000, 1)
p = np.poly1d(z)
x_line = np.linspace(df['Experience'].min(), df['Experience'].max(), 100)
ax2.plot(x_line, p(x_line), 'k--', linewidth=2, label='Trend')
ax2.set_title("Viz 2: Salary Growth with Experience", fontweight='bold')
ax2.set_xlabel("Years of Experience")
ax2.set_ylabel("Salary (₹ Lakhs)")
ax2.legend(fontsize=7, loc='upper left')

# ── VIZ 3: Average Salary by Location (Bar Chart)
ax3 = fig.add_subplot(gs[1, 0])
loc_avg = df.groupby('Location')['Salary'].mean().sort_values(ascending=False) / 100000
bars = ax3.bar(loc_avg.index, loc_avg.values, color=sns.color_palette("husl", len(loc_avg)))
ax3.set_title("Viz 3: Average Salary by Location", fontweight='bold')
ax3.set_xlabel("Location")
ax3.set_ylabel("Avg Salary (₹ Lakhs)")
for bar, val in zip(bars, loc_avg.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'₹{val:.1f}L', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ── VIZ 4: Performance Score vs Salary Increment (Bar + Line)
ax4 = fig.add_subplot(gs[1, 1])
perf_data = df.groupby('Performance_Score').agg(
    Avg_Increment=('Salary_Increment', 'mean'),
    Avg_Salary=('Salary', 'mean')
).reset_index()
ax4b = ax4.twinx()
bars2 = ax4.bar(perf_data['Performance_Score'], perf_data['Avg_Increment'],
                color='#3498db', alpha=0.7, label='Avg Increment (%)')
ax4b.plot(perf_data['Performance_Score'], perf_data['Avg_Salary']/100000,
          'ro-', linewidth=2, markersize=8, label='Avg Salary (L)')
ax4.set_title("Viz 4: Performance Score vs Salary & Increment", fontweight='bold')
ax4.set_xlabel("Performance Score (1=Low, 5=High)")
ax4.set_ylabel("Avg Salary Increment (%)", color='#3498db')
ax4b.set_ylabel("Avg Salary (₹ Lakhs)", color='red')
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4b.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

plt.savefig('hr_salary_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Dashboard saved as hr_salary_dashboard.png")

# ─────────────────────────────────────────────
# 6. LINEAR REGRESSION MODEL
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: LINEAR REGRESSION MODEL")
print("="*60)

le = LabelEncoder()
df_model = df.copy()
for col in ['Designation', 'Education', 'Location']:
    df_model[col + '_enc'] = le.fit_transform(df_model[col])

features = ['Experience', 'Performance_Score', 'Designation_enc', 'Education_enc', 'Location_enc']
X = df_model[features]
y = df_model['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation Results:")
print(f"  R² Score          : {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"  MAE               : ₹{mae:,.0f}")
print(f"  RMSE              : ₹{rmse:,.0f}")

print(f"\nFeature Coefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:<22}: ₹{coef:>12,.2f}")
print(f"  {'Intercept':<22}: ₹{model.intercept_:>12,.2f}")

# Actual vs Predicted Plot
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Linear Regression — Model Evaluation", fontweight='bold')

axes[0].scatter(y_test/100000, y_pred/100000, alpha=0.5, color='steelblue', s=30)
min_val = min(y_test.min(), y_pred.min()) / 100000
max_val = max(y_test.max(), y_pred.max()) / 100000
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
axes[0].set_xlabel("Actual Salary (₹ Lakhs)")
axes[0].set_ylabel("Predicted Salary (₹ Lakhs)")
axes[0].set_title(f"Actual vs Predicted  (R²={r2:.3f})")

residuals = y_test - y_pred
axes[1].hist(residuals/100000, bins=30, color='coral', edgecolor='white')
axes[1].axvline(0, color='black', linestyle='--')
axes[1].set_xlabel("Residual (₹ Lakhs)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plt.savefig('hr_model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Model evaluation plot saved as hr_model_evaluation.png")

# ─────────────────────────────────────────────
# 7. KEY INSIGHTS & ACTION PLAN
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6: KEY INSIGHTS")
print("="*60)

insights = [
    ("1", "Salary scales significantly with designation",
     f"Directors earn ~{df[df['Designation']=='Director']['Salary'].mean()/df[df['Designation']=='Junior Developer']['Salary'].mean():.1f}x more than Junior Developers, confirming a structured pay hierarchy."),
    ("2", "Experience drives salary growth",
     f"Each additional year of experience adds approximately ₹{model.coef_[0]:,.0f} to salary (per regression coefficient)."),
    ("3", "Bangalore & Mumbai pay a premium",
     f"Bangalore avg: ₹{df[df['Location']=='Bangalore']['Salary'].mean()/100000:.1f}L vs Chennai avg: ₹{df[df['Location']=='Chennai']['Salary'].mean()/100000:.1f}L — a {((df[df['Location']=='Bangalore']['Salary'].mean()/df[df['Location']=='Chennai']['Salary'].mean())-1)*100:.1f}% gap."),
    ("4", "High performers receive better increments",
     f"Score-5 employees get avg {df[df['Performance_Score']==5]['Salary_Increment'].mean():.1f}% increment vs {df[df['Performance_Score']==1]['Salary_Increment'].mean():.1f}% for score-1 employees."),
    ("5", "Compensation anomalies exist",
     f"{len(outliers)} employees ({len(outliers)/n*100:.1f}%) have salaries deviating >2 std from role peers — potential pay equity issues."),
    ("6", "Education adds a modest premium",
     f"PhD holders earn ~{(edu_multiplier['PhD']-1)*100:.0f}% more than Bachelor holders at the same role and experience level."),
    ("7", "Linear Regression explains salary well",
     f"R²={r2:.3f} — the model explains {r2*100:.1f}% of salary variance using role, experience, location, education, and performance."),
]

for num, title, detail in insights:
    print(f"\n  Insight {num}: {title}")
    print(f"  → {detail}")

print("\n" + "="*60)
print("STEP 7: PRACTICAL ACTION PLAN")
print("="*60)

actions = [
    ("Audit outlier employees", f"Review the {len(outliers)} flagged employees for unjustified pay gaps. Correct underpaid cases to reduce attrition risk."),
    ("Standardise location allowances", "Formalise a transparent location-based pay band policy to justify the Bangalore/Mumbai premium."),
    ("Link increments to performance rigorously", "Ensure all Score-4/5 employees receive ≥10% increment; Score-1/2 employees should receive structured PIP before increment."),
    ("Create experience-based pay bands", "Define salary bands per designation per experience bracket (0-3, 3-6, 6-10 yrs) to prevent compression."),
    ("Education-based joining bonus", "Offer structured joining bonuses for Master/PhD hires rather than inflating base salary to maintain pay equity."),
]

for i, (action, detail) in enumerate(actions, 1):
    print(f"\n  Action {i}: {action}")
    print(f"  → {detail}")

print("\n" + "="*60)
print("[DONE] ANALYSIS COMPLETE")
print("   Files generated:")
print("   • hr_salary_data.csv")
print("   • hr_salary_dashboard.png")
print("   • hr_model_evaluation.png")
print("="*60)
