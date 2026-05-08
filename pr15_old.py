import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.linear_model import LogisticRegression

# hide unnecessary warnings
warnings.filterwarnings("ignore")

# ensure directory for plots exists
os.makedirs("figures", exist_ok=True)

# =============================================================================
# styling & data loading
# =============================================================================

# apply ieee formatting for professional charts
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# load dataset and filter for english content
df = pd.read_csv("multi_platform_social_sentiment_evolution.csv")
df = df[df['language'] == 'English']

# clean numeric columns and drop missing values
numeric_cols = ['likes', 'comments', 'shares', 'views', 'toxicity_score', 'followers', 'account_age_days', 'hours_since_post']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=[col for col in numeric_cols if col in df.columns])

# add binary flag for posts with at least one comment
df['has_comments'] = (df['comments'] > 0).astype(int)

# pick the top 6 platforms by volume for analysis
top_6_platforms = df['platform'].value_counts().nlargest(6).index.tolist()
df_top6 = df[df['platform'].isin(top_6_platforms)]

# =============================================================================
# task 7: low activity users (merged from first script)
# =============================================================================

# count posts per user and platform
user_activity = df.groupby(['user_id', 'platform', 'location'], observed=True).size().reset_index(name='post_count')

# filter for users with fewer than 5 posts
user_activity_low = user_activity[user_activity['post_count'] < 5]
chart_data = user_activity_low.groupby(['location', 'platform']).size().reset_index(name='user_count')

# print summary table
table_data = chart_data.pivot(index='location', columns='platform', values='user_count').fillna(0).astype(int)
print("\n=== task 7: users with < 5 posts by location ===")
print(table_data.to_string())

# plot location-based activity
plt.figure(figsize=(7.16, 3.5))
sns.barplot(data=chart_data, x='location', y='user_count', hue='platform', palette='viridis', edgecolor='black')
plt.xlabel('geographical location')
plt.ylabel('number of users (< 5 posts)')
plt.legend(title='platform', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6, frameon=False)
plt.xticks(rotation=45, ha='right')
plt.savefig("figures/task7_ieee_location_platform.png")
plt.close()

# =============================================================================
# task 2: logistic regression for comments
# =============================================================================

print("\n=== task 2: probability of comments by platform ===")
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
axes2 = axes2.flatten()

for i, plat in enumerate(top_6_platforms):
    plat_df = df_top6[df_top6['platform'] == plat]
    X = plat_df[['likes', 'hours_since_post']].values
    y = plat_df['has_comments'].values
    
    if len(np.unique(y)) > 1:
        # train model to predict if a post gets a comment
        model = LogisticRegression().fit(X, y)
        
        # create grid for heatmap visualization
        likes_grid = np.geomspace(1, max(plat_df['likes'].quantile(0.95), 10), 50)
        hours_grid = np.linspace(0.1, plat_df['hours_since_post'].quantile(0.95), 50)
        likes_mesh, hours_mesh = np.meshgrid(likes_grid, hours_grid)
        
        # predict probabilities across the grid
        prob_grid = model.predict_proba(np.column_stack([likes_mesh.ravel(), hours_mesh.ravel()]))[:, 1].reshape(likes_mesh.shape)
        
        # plot probability contours
        contour = axes2[i].contourf(likes_mesh, hours_mesh, prob_grid, levels=20, cmap='YlOrRd')
        fig2.colorbar(contour, ax=axes2[i], label='P(comment > 0)')
        
        # find likes needed for 50% chance of comment at 12 hours
        x_test = np.column_stack([likes_grid, np.full_like(likes_grid, 12)])
        probs = model.predict_proba(x_test)[:, 1]
        idx = np.where(probs >= 0.5)[0]
        result = f"{likes_grid[idx[0]]:.0f} likes" if len(idx) > 0 else "never"
        print(f"platform: {plat.ljust(12)} | 50% prob at 12h: {result}")

    axes2[i].set_xscale('log')
    axes2[i].set_title(plat)

plt.tight_layout()
plt.savefig("figures/task2_comment_probability_6platforms.png", dpi=150)
plt.close()

# =============================================================================
# task 3: account age & popularity impact
# =============================================================================

# bin account age and follower counts into categories
df_top6['age_class'] = pd.cut(df_top6['account_age_days'], 
                             bins=[0, 365, 1095, 1825, 2920, 3650], 
                             labels=['youth', 'teenager', 'junior', 'senior', 'eldest'])

df_top6['popularity'] = pd.cut(df_top6['followers'], 
                              bins=[-1, 1000, 10000, 100000, np.inf], 
                              labels=['nano', 'micro', 'macro', 'mega'])

fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
axes3 = axes3.flatten()

for i, plat in enumerate(top_6_platforms):
    # visualize how age and popularity correlate with engagement
    sns.scatterplot(data=df_top6[df_top6['platform'] == plat], x='likes', y='comments', 
                    hue='age_class', size='popularity', sizes=(20, 150), alpha=0.5, ax=axes3[i])
    axes3[i].set_title(plat)

plt.tight_layout()
plt.savefig("figures/task3_age_followers_6platforms.png", dpi=150)
plt.close()

# =============================================================================
# task 4: toxicity impact
# =============================================================================

# filter for posts in the top 25% of toxicity scores
tox_thresh = df['toxicity_score'].quantile(0.75)
high_tox_df = df_top6[df_top6['toxicity_score'] > tox_thresh]

fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))
axes4 = axes4.flatten()

for i, plat in enumerate(top_6_platforms):
    plat_tox = high_tox_df[high_tox_df['platform'] == plat]
    
    if len(plat_tox) > 10 and len(np.unique(plat_tox['has_comments'])) > 1:
        # model comment probability specifically for high toxicity posts
        model_tox = LogisticRegression().fit(plat_tox[['likes', 'hours_since_post']].values, plat_tox['has_comments'].values)
        
        likes_grid_t = np.geomspace(1, max(plat_tox['likes'].quantile(0.95), 10), 50)
        hours_grid_t = np.linspace(0.1, plat_tox['hours_since_post'].quantile(0.95), 50)
        lm, hm = np.meshgrid(likes_grid_t, hours_grid_t)
        
        # generate toxic post probability map
        pg = model_tox.predict_proba(np.column_stack([lm.ravel(), hm.ravel()]))[:, 1].reshape(lm.shape)
        axes4[i].contourf(lm, hm, pg, levels=20, cmap='Reds')

    axes4[i].set_xscale('log')
    axes4[i].set_title(f'{plat} (high tox)')

plt.tight_layout()
plt.savefig("figures/task4_toxicity_6platforms.png", dpi=150)
plt.close()
