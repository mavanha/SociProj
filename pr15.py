# import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
import os
import warnings

# suppress warnings
warnings.filterwarnings("ignore")

# create output directory for figures
os.makedirs("figures", exist_ok=True)

# configure plotting style
sns.set_theme(style="whitegrid") 
sns.set_context("notebook", font_scale=1.25) 
plt.rcParams.update({
    'patch.linewidth': 0.0,
})

# =============================================================================
# 1. load data & initial setup
# =============================================================================
# load dataset and filter for english language
df = pd.read_csv("multi_platform_social_sentiment_evolution.csv")
df = df[df['language'] == 'English']

# convert target columns to numeric values
numeric_cols = ['likes', 'comments', 'shares', 'views', 'toxicity_score', 'followers', 'account_age_days']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# identify missing columns
missing = [c for c in ['likes', 'comments', 'hours_since_post', 'platform'] if c not in df.columns]

# drop rows missing essential metrics and parse dates
df = df.dropna(subset=['likes', 'comments', 'hours_since_post', 'platform'])
df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
df['has_comments'] = (df['comments'] > 0).astype(int)

# get unique platform names
platforms = df['platform'].dropna().unique()

# --- summary statistics table ---
# calculate high-level dataset statistics
total_users = df['user_id'].nunique() if 'user_id' in df.columns else len(df)
total_posts = len(df)
time_min = df['timestamp_dt'].min()
time_max = df['timestamp_dt'].max()

print("\n--- summary statistics ---")
summary_data = {
    "metric": ["total users", "total posts", "timeframe start", "timeframe end", "duration"],
    "value": [total_users, total_posts, time_min.strftime('%Y-%m-%d'), time_max.strftime('%Y-%m-%d'), "6 months"]
}
print(pd.DataFrame(summary_data).to_string(index=False))

# --- platform distribution chart ---
# calculate and plot platform distribution pie chart
platform_counts = df['platform'].value_counts()
platform_pct = (platform_counts / len(df) * 100).round(1)

print("\n--- platform distribution (data) ---")
platform_df = pd.DataFrame({'count': platform_counts, 'percentage': platform_pct.astype(str) + '%'})
print(platform_df.to_string())

plt.figure(figsize=(7, 7))
plt.pie(platform_counts, labels=platform_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set3'))
#plt.title('platform distribution (volume of dataset)')
plt.tight_layout()
plt.savefig("figures/00_platform_distribution_pie.png", dpi=150)
plt.close()

# --- demographic breakdown (age classes) ---
# categorize users by account age
bins = [0, 365, 1095, 1825, 2920, 3650]
labels = ['youth', 'teenager', 'junior', 'senior', 'eldest']
if 'account_age_days' in df.columns:
    df['age_class'] = pd.cut(df['account_age_days'], bins=bins, labels=labels)
    print("\n--- demographic breakdown (age classes) ---")
    print(df['age_class'].value_counts().reindex(labels).to_string())

# --- geographic data concentration ---
# print geographical locations if available
if 'location' in df.columns:
    loc_counts = df['location'].value_counts()
    print("\n--- geographic data concentration ---")
    print(loc_counts.to_string())

# =============================================================================
# task 1: sentiment vs engagement — Pearson correlation + p-values
# =============================================================================
# find correlations between sentiment and engagement metrics
sentiment_cols = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
engagement_cols = ['likes', 'comments', 'shares', 'views']
available_sents = [c for c in sentiment_cols if c in df.columns]

if available_sents:
    corr_results = []
    for eng in engagement_cols:
        for sent in available_sents:
            mask = df[[eng, sent]].notna().all(axis=1)
            if mask.sum() > 2:
                r, p = stats.pearsonr(df.loc[mask, eng], df.loc[mask, sent])
                corr_results.append({'engagement': eng, 'sentiment': sent, 'pearson_r': round(r, 4), 'p_value': p})

    print("\n=== task 1: correlation table ===")
    print(pd.DataFrame(corr_results).to_string(index=False))

    # plot correlation heatmap
    corr_matrix = df[engagement_cols + available_sents].corr()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f')
    #plt.title('sentiment vs engagement: Pearson correlation')
    plt.tight_layout()
    plt.savefig("figures/task1_correlation_heatmap.png", dpi=150)
    plt.close()

# logistic regression function to find likes threshold for 50% comment probability
def get_likes_needed_for_50(sub_df, hours_target=12):
    if len(sub_df) < 30 or len(sub_df['has_comments'].unique()) < 2:
        return np.nan, None
    
    X = sub_df[['likes', 'hours_since_post']].values
    y = sub_df['has_comments'].values
    
    model = LogisticRegression()
    model.fit(X, y)
    
    max_l = max(sub_df['likes'].quantile(0.95), 10)
    likes_test = np.geomspace(1, max_l, 200)
    probs = model.predict_proba(np.column_stack([likes_test, np.full_like(likes_test, hours_target)]))[:, 1]
    
    idx = np.where(probs >= 0.5)[0]
    return (likes_test[idx[0]], model) if len(idx) > 0 else (np.nan, model)

# =============================================================================
# task 2: how many likes are required to generate comments
# =============================================================================
# plot comment probability curves by platform
print("\n=== task 2: likes needed for 50% P(comment>0) ===")

plt.figure(figsize=(10, 6))
colors = sns.color_palette("tab10", len(platforms))

for i, plat in enumerate(platforms):
    plat_df = df[df['platform'] == plat]
    print(f"\n[{plat.capitalize()}] at different hours since post:")
    
    for h in [0.5, 1, 6, 12, 24]:
        likes_req, model = get_likes_needed_for_50(plat_df, hours_target=h)
        print(f"  hours={h}: {likes_req:.0f} likes" if not np.isnan(likes_req) else f"  hours={h}: never reaches 50%")
    
    _, model = get_likes_needed_for_50(plat_df, hours_target=12)
    if model is not None:
        likes_plot = np.geomspace(1, max(plat_df['likes'].quantile(0.95), 10), 100)
        probs_plot = model.predict_proba(np.column_stack([likes_plot, np.full_like(likes_plot, 12)]))[:, 1]
        plt.plot(likes_plot, probs_plot, label=f"{plat} (at 12h)", color=colors[i], linewidth=2)

plt.axhline(0.5, color='black', linestyle='--', label='50% threshold')
plt.xscale('log')
plt.xlabel('likes (log scale)')
plt.ylabel('P(comment > 0)')
#plt.title('probability of generating comments by platform (at 12 hours)')
plt.legend()
plt.tight_layout()
plt.savefig("figures/task2_comment_prob_multiplatform.png", dpi=150)
plt.close()

# =============================================================================
# task 3: age range & followers impact
# =============================================================================
# analyze required likes grouped by user age class
print("\n=== task 3: impact of age & popularity per platform ===")
task3_age_data = []

print("\n--- likes needed for 50% P(comment>0) by age class (at 12h) ---")
for plat in platforms:
    print(f"\n[{plat.capitalize()}]")
    for age_cls in labels:
        sub_df = df[(df['platform'] == plat) & (df['age_class'] == age_cls)]
        likes_req, _ = get_likes_needed_for_50(sub_df, 12)
        
        task3_age_data.append({'platform': plat, 'age class': age_cls, 'likes needed': likes_req})
        if not np.isnan(likes_req):
            print(f"  {age_cls}: {likes_req:.0f} likes needed")
        else:
            reason = "insufficient data" if len(sub_df) < 30 else "never reaches 50%"
            print(f"  {age_cls}: {reason}")

task3_age_df = pd.DataFrame(task3_age_data).dropna()
if not task3_age_df.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=task3_age_df, x='age class', y='likes needed', hue='platform')
    #plt.title('likes needed for 50% comment prob by age class & platform (12h)')
    plt.ylabel('likes')
    plt.tight_layout()
    plt.savefig("figures/task3_age_multiplatform.png", dpi=150)
    plt.close()

# categorize users by follower count
pop_bins = [-1, 1000, 10000, 100000, np.inf]
pop_labels = ['nano (<1k)', 'micro (1k-10k)', 'macro (10k-100k)', 'mega (>100k)']
df['popularity'] = pd.cut(df['followers'], bins=pop_bins, labels=pop_labels)

# analyze required likes grouped by popularity
print("\n--- likes needed for 50% P(comment>0) by popularity (at 12h) ---")
for plat in platforms:
    print(f"\n[{plat.capitalize()}]")
    for pop_cls in pop_labels:
        sub_df = df[(df['platform'] == plat) & (df['popularity'] == pop_cls)]
        likes_req, _ = get_likes_needed_for_50(sub_df, 12)
        
        if not np.isnan(likes_req):
            print(f"  {pop_cls}: {likes_req:.0f} likes needed")
        else:
            reason = "insufficient data" if len(sub_df) < 30 else "never reaches 50%"
            print(f"  {pop_cls}: {reason}")

# =============================================================================
# task 4: high toxicity impact
# =============================================================================
# check engagement probabilities specifically for highly toxic posts
if 'toxicity_score' in df.columns:
    print("\n=== task 4: high toxicity impact ===")
    tox_data = []
    
    for plat in platforms:
        plat_df = df[df['platform'] == plat]
        toxicity_thresh = plat_df['toxicity_score'].quantile(0.75)
        high_tox = plat_df[plat_df['toxicity_score'] > toxicity_thresh]
        
        print(f"\n[{plat.capitalize()}] high tox thresh: {toxicity_thresh:.2f}")
        for h in [0.5, 1, 6, 12, 24]:
            likes_req, _ = get_likes_needed_for_50(high_tox, h)
            if h == 12: tox_data.append({'platform': plat, 'likes needed (12h)': likes_req})
            
            print(f"  hours={h}: {likes_req:.0f} likes" if not np.isnan(likes_req) else f"  hours={h}: never reaches 50%")
            
    tox_df = pd.DataFrame(tox_data).dropna()
    if not tox_df.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=tox_df, x='platform', y='likes needed (12h)', palette='Reds_r')
        #plt.title('likes needed for 50% comment prob on high toxicity posts (12h)')
        plt.ylabel('likes')
        plt.tight_layout()
        plt.savefig("figures/task4_toxicity_multiplatform.png", dpi=150)
        plt.close()

# =============================================================================
# task 5: geographical locations
# =============================================================================
# analyze and map comment thresholds by user location
if 'location' in df.columns:
    print("\n=== task 5: location impact ===")
    locations = df['location'].unique()
    loc_matrix = pd.DataFrame(index=locations, columns=platforms, dtype=float)
    
    print("\n--- likes needed for 50% P(comment>0) by location (at 12h) ---")
    for plat in platforms:
        print(f"\n[{plat.capitalize()}]")
        for loc in locations:
            sub_df = df[(df['platform'] == plat) & (df['location'] == loc)]
            likes_req, _ = get_likes_needed_for_50(sub_df, 12)
            
            if not np.isnan(likes_req):
                loc_matrix.at[loc, plat] = likes_req
                print(f"  {loc}: {likes_req:.0f} likes needed")
            else:
                reason = "insufficient data" if len(sub_df) < 30 else "never reaches 50%"
                print(f"  {loc}: {reason}")
                
    loc_matrix = loc_matrix.dropna(how='all')
    if not loc_matrix.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(loc_matrix, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'likes needed for 50% P(comment)'})
        #plt.title('likes needed for 50% comment probability by location and platform (12h)')
        plt.ylabel('location')
        plt.xlabel('platform')
        plt.tight_layout()
        plt.savefig("figures/task5_location_heatmap.png", dpi=150)
        plt.close()

# =============================================================================
# Task 6: Base histogram & cumulative distribution (per platform)
# =============================================================================
# bin users by total post count
user_counts = df.groupby('user_id').size().reset_index(name='post_count')
user_counts['post_bin'] = pd.cut(user_counts['post_count'],
                                  bins=[0, 5, 10, 15, 20, 25, np.inf],
                                  labels=['<5', '5-10', '10-15', '15-20', '20-25', '25+'], right=False)

if 'post_bin' in df.columns:
    df.drop(columns=['post_bin'], inplace=True)
df = df.merge(user_counts[['user_id', 'post_bin']], on='user_id')

task6_data = df.groupby(['post_bin', 'platform'], observed=False).size().unstack(fill_value=0)

print("\n--- task 6: binned posts per platform (stacked chart data) ---")
print(task6_data.to_string())

# calculate and print cumulative distribution of user posts
print("\n--- task 6: cumulative distribution (A_k) per platform ---")
platforms = df['platform'].unique()

for plat in platforms:
    plat_df = df[df['platform'] == plat]
    plat_k_counts = plat_df.groupby('user_id').size().value_counts().sort_index()
    plat_A_k = plat_k_counts[::-1].cumsum()[::-1]
    
    print(f"\n=== {plat} ===")
    print("k (posts) | users with >= k posts")
    print(plat_A_k.head(15).to_string())

# plot: stacked bar of user activity
plt.figure(figsize=(10, 6))
task6_data.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='viridis', edgecolor='white')
#plt.title('User activity distribution (posts color-coded by platform)')
plt.xlabel('User activity level (total posts per user)')
plt.ylabel('Total number of posts')
plt.legend(title='Platform', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("figures/task6_user_posts_platform_stacked.png", dpi=150)
plt.close()

# plot: cumulative distribution (A_k) per platform
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(platforms)))

for plat, color in zip(platforms, colors):
    plat_df = df[df['platform'] == plat]
    plat_k_counts = plat_df.groupby('user_id').size().value_counts().sort_index()
    plat_A_k = plat_k_counts[::-1].cumsum()[::-1]
    plt.plot(plat_A_k.index, plat_A_k.values, marker='o', linestyle='-', color=color, label=plat, alpha=0.8)

plt.xscale('log')
plt.yscale('log')
#plt.title('Cumulative distribution of posts per user (by platform)')
plt.xlabel('Number of posts (k)')
plt.ylabel('Users with >= k posts (A_k)')
plt.legend(title='Platform')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("figures/task6_user_posts_cumulative_platform.png", dpi=150)
plt.close()


# =============================================================================
# Task 7: Adding geography
# =============================================================================
# analyze post bins mapped with location data
location_bins = df.groupby(['location', 'post_bin'], observed=True).size().unstack(fill_value=0)

print("\n--- task 7: distribution of posts per user by location ---")
print(location_bins.to_string())

# plot subplots of user activity by location
g = sns.catplot(
    data=df, x='post_bin', hue='platform', col='location', kind='count', 
    col_wrap=4, # Change to 4 to make it wider and only 2 rows deep
    height=3,   # Reduce height to save vertical space
    aspect=1.0, # Make subplots more square
    palette='viridis'
)
# move legend and adjust titles
sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, title='Platform')
plt.tight_layout()
g.fig.suptitle('User activity by platform across different locations', y=1.02)
g.set_axis_labels("User activity level (bins)", "Total posts")
g.set_titles("{col_name}")
plt.savefig("figures/task7_location_platform_subplots.png", dpi=150)
plt.close()


# =============================================================================
# Task 8: High toxicity focus
# =============================================================================
# filter and analyze only the top 10% most toxic posts
tox_threshold = df['toxicity_score'].quantile(0.90)
high_tox = df[df['toxicity_score'] > tox_threshold].copy()

tox_user_counts = high_tox.groupby('user_id').size().reset_index(name='post_count')
tox_user_counts['post_bin'] = pd.cut(tox_user_counts['post_count'],
                                      bins=[0, 5, 10, 15, 20, 25, np.inf],
                                      labels=['<5', '5-10', '10-15', '15-20', '20-25', '25+'], right=False)

if 'post_bin' in high_tox.columns:
    high_tox.drop(columns=['post_bin'], inplace=True)
high_tox = high_tox.merge(tox_user_counts[['user_id', 'post_bin']], on='user_id')
tox_platform_data = high_tox.groupby(['post_bin', 'platform'], observed=False).size().unstack(fill_value=0)

print("\n--- task 8: binned toxic posts per platform (stacked chart data) ---")
print(tox_platform_data.to_string())

print("\n--- task 8: cumulative distribution (A_k) per platform - high toxicity ---")
for plat in platforms:
    plat_df = high_tox[high_tox['platform'] == plat]
    if not plat_df.empty:
        plat_k_counts = plat_df.groupby('user_id').size().value_counts().sort_index()
        plat_A_k = plat_k_counts[::-1].cumsum()[::-1]
        
        print(f"\n=== {plat} (high toxicity) ===")
        print("k (posts) | users with >= k posts")
        print(plat_A_k.head(10).to_string())
    else:
        print(f"\n=== {plat} (high toxicity) ===")
        print("No highly toxic posts for this platform.")

# plot: stacked bar of high toxicity activity
plt.figure(figsize=(10, 6))
tox_platform_data.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='Reds', edgecolor='white')
#plt.title('High toxicity posts per user (top 10% toxicity, by platform)')
plt.xlabel('Toxic activity level (toxic posts per user)')
plt.ylabel('Total number of toxic posts')
plt.legend(title='Platform', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("figures/task8_toxicity_platform_stacked.png", dpi=150)
plt.close()

# plot: A_k per platform for high toxicity posts
plt.figure(figsize=(10, 6))
for plat, color in zip(platforms, colors):
    plat_df = high_tox[high_tox['platform'] == plat]
    if not plat_df.empty:
        plat_k_counts = plat_df.groupby('user_id').size().value_counts().sort_index()
        plat_A_k = plat_k_counts[::-1].cumsum()[::-1]
        plt.plot(plat_A_k.index, plat_A_k.values, marker='o', linestyle='-', color=color, label=plat, alpha=0.8)

plt.xscale('log')
plt.yscale('log')
#plt.title('Cumulative distribution of toxic posts per user (by platform)')
plt.xlabel('Number of toxic posts (k)')
plt.ylabel('Users with >= k posts (A_k)')
plt.legend(title='Platform')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("figures/task8_toxicity_cumulative_platform.png", dpi=150)
plt.close()


# =============================================================================
# Task 9: High engagement focus
# =============================================================================
# filter and analyze highly engaging (viral) posts
likes_thresh = df['likes'].quantile(0.75)
comments_thresh = df['comments'].quantile(0.75)
high_engage = df[(df['likes'] > likes_thresh) & (df['comments'] > comments_thresh)].copy()

engage_user_counts = high_engage.groupby('user_id').size().reset_index(name='post_count')
engage_user_counts['post_bin'] = pd.cut(engage_user_counts['post_count'],
                                         bins=[0, 5, 10, 15, 20, 25, np.inf],
                                         labels=['<5', '5-10', '10-15', '15-20', '20-25', '25+'], right=False)

if 'post_bin' in high_engage.columns:
    high_engage.drop(columns=['post_bin'], inplace=True)
high_engage = high_engage.merge(engage_user_counts[['user_id', 'post_bin']], on='user_id')

engage_platform_data = high_engage.groupby(['post_bin', 'platform'], observed=False).size().unstack(fill_value=0)

# compare high engagement users to overall averages
print("\n--- task 9: high-engagement users vs overall ---")
print(f"  avg followers (high-eng): {high_engage['followers'].mean():.0f} vs overall: {df['followers'].mean():.0f}")
print(f"  avg account age (high-eng): {high_engage['account_age_days'].mean():.0f} vs overall: {df['account_age_days'].mean():.0f}")
print(f"  avg toxicity (high-eng): {high_engage['toxicity_score'].mean():.2f} vs overall: {df['toxicity_score'].mean():.2f}")
print(f"  positive sentiment (high-eng): {high_engage['sentiment_positive'].mean():.3f} vs overall: {df['sentiment_positive'].mean():.3f}")

print("\n--- task 9: binned viral posts per platform (stacked chart data) ---")
print(engage_platform_data.to_string())

print("\n--- task 9: cumulative distribution (A_k) per platform - high engagement ---")
for plat in platforms:
    plat_df = high_engage[high_engage['platform'] == plat]
    if not plat_df.empty:
        plat_k_counts = plat_df.groupby('user_id').size().value_counts().sort_index()
        plat_A_k = plat_k_counts[::-1].cumsum()[::-1]
        
        print(f"\n=== {plat} (high engagement) ===")
        print("k (posts) | users with >= k posts")
        print(plat_A_k.head(10).to_string())
    else:
        print(f"\n=== {plat} (high engagement) ===")
        print("No viral posts for this platform.")

# plot: stacked bar of viral post distribution
plt.figure(figsize=(10, 6))
engage_platform_data.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='Greens', edgecolor='white')
#plt.title('Viral engagement posts per user (by platform)')
plt.xlabel('Viral activity level (viral posts per user)')
plt.ylabel('Total number of viral posts')
plt.legend(title='Platform', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("figures/task9_engage_platform_stacked.png", dpi=150)
plt.close()

# plot: A_k per platform for high engagement posts
plt.figure(figsize=(10, 6))
for plat, color in zip(platforms, colors):
    plat_df = high_engage[high_engage['platform'] == plat]
    if not plat_df.empty:
        plat_k_counts = plat_df.groupby('user_id').size().value_counts().sort_index()
        plat_A_k = plat_k_counts[::-1].cumsum()[::-1]
        plt.plot(plat_A_k.index, plat_A_k.values, marker='o', linestyle='-', color=color, label=plat, alpha=0.8)

plt.xscale('log')
plt.yscale('log')
#plt.title('Cumulative distribution of viral posts per user (by platform)')
plt.xlabel('Number of viral posts (k)')
plt.ylabel('Users with >= k posts (A_k)')
plt.legend(title='Platform')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("figures/task9_engage_cumulative_platform.png", dpi=150)
plt.close()


# =============================================================================
# task 10: propagation speed — using two valid interpretations (per platform)
# =============================================================================

# -----------------------------------------------------------------------------
# interpretation 1: posting frequency (speed of post propagation by user)
# -----------------------------------------------------------------------------
# calculate time difference between consecutive posts per user
df_sorted = df.sort_values(['platform', 'user_id', 'timestamp_dt']).copy()
df_sorted['time_diff_hours'] = df_sorted.groupby(['platform', 'user_id'])['timestamp_dt'].diff().dt.total_seconds() / 3600

df_valid = df_sorted.dropna(subset=['time_diff_hours']).copy()
df_valid = df_valid[df_valid['time_diff_hours'] > 0]
df_valid['posting_speed'] = 1 / df_valid['time_diff_hours']

print("\n--- task 10a: propagation speed (posts per hour) by platform ---")
stats_post = df_valid.groupby('platform')['posting_speed'].agg(['mean', 'std', 'median']).reset_index()
print(stats_post.to_string(index=False, float_format="%.4f"))

# plot posting speed boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_valid, x='platform', y='posting_speed', palette='Set3')
plt.yscale('log')
#plt.title('Propagation speed distribution by platform (posts per hour)')
plt.xlabel('Platform')
plt.ylabel('Posts per hour (log scale)')
plt.tight_layout()
plt.savefig("figures/task10_posting_speed_per_platform.png", dpi=150) 
plt.close()


# -----------------------------------------------------------------------------
# interpretation 2: engagement velocity (speed of engagement propagation)
# -----------------------------------------------------------------------------
# calculate engagement divided by hours active to get velocity
df['engagement_speed'] = df['total_engagement'] / df['hours_since_post'].replace(0, np.nan)
df['engagement_speed'] = df['engagement_speed'].replace([np.inf, -np.inf], np.nan)
df_prop = df.dropna(subset=['engagement_speed', 'platform'])

user_avg_speeds_df = df_prop.groupby(['platform', 'user_id'])['engagement_speed'].mean().reset_index()
user_avg_speeds_df = user_avg_speeds_df.dropna(subset=['engagement_speed'])

print("\n--- task 10b: propagation speed (avg engagement/hour per user) by platform ---")
stats_eng = user_avg_speeds_df.groupby('platform')['engagement_speed'].agg(['mean', 'std', 'median']).reset_index()
print(stats_eng.to_string(index=False, float_format="%.2f"))

# plot engagement velocity boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=user_avg_speeds_df, x='platform', y='engagement_speed', palette='Set3')
plt.yscale('log')
#plt.title('Average engagement velocity per user by platform')
plt.xlabel('Platform')
plt.ylabel('Avg engagement per hour (log scale)')
plt.tight_layout()
plt.savefig("figures/task10_engagement_velocity_per_platform.png", dpi=150) 
plt.close()

print("\nanalysis complete. all figures saved to: figures/")