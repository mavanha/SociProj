# Multi-Platform Social Sentiment Evolution

## 📊 Dataset Overview

This comprehensive social media dataset captures sentiment evolution across 6 major platforms over 6 months. It includes 150,000 posts with detailed engagement metrics, sentiment analysis, virality indicators, and cross-platform propagation patterns.

**Dataset Size**: 150,000 posts  
**Time Period**: April 2025 - October 2025 (6 months)  
**Platforms**: Twitter/X, Reddit, Instagram, YouTube, TikTok, Facebook  
**Total Features**: 31 columns

## 🎯 Use Cases

- **Sentiment Analysis**: Track how sentiment evolves across different topics and platforms
- **Virality Prediction**: Build models to predict which content will go viral
- **Platform Comparison**: Compare engagement patterns across social media platforms
- **Topic Modeling**: Identify trending topics and sentiment patterns
- **Influence Analysis**: Study the relationship between follower count and engagement
- **Cross-Platform Studies**: Analyze how content spreads across different platforms
- **Temporal Patterns**: Discover optimal posting times and day-of-week effects
- **Language Analysis**: Compare sentiment across different languages

## 📁 Dataset Structure

### Post Identification
- `post_id`: Unique identifier for each post (format: PLATFORM_DATE_SEQUENCE)
- `platform`: Social media platform (Twitter, Reddit, Instagram, YouTube, TikTok, Facebook)
- `timestamp`: Exact post timestamp (YYYY-MM-DD HH:MM:SS)
- `date`: Post date (YYYY-MM-DD)

### Temporal Features
- `hour_of_day`: Hour when posted (0-23)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `is_weekend`: Binary indicator for weekend posts

### User Metrics
- `user_id`: Anonymized user identifier
- `followers`: Number of followers/subscribers
- `account_age_days`: Age of account in days
- `verified`: Binary indicator for verified accounts

### Content Characteristics
- `topic`: Primary topic/category (15 topics including Technology, Politics, Entertainment, etc.)
- `language`: Language of content (10 languages)
- `content_length`: Character count of content
- `media_type`: Type of media (Text, Image, Video, Link, Poll)
- `num_hashtags`: Number of hashtags used

### Sentiment Analysis
- `sentiment_category`: Overall sentiment (Positive, Negative, Neutral)
- `sentiment_positive`: Positive sentiment score (0-1)
- `sentiment_negative`: Negative sentiment score (0-1)
- `sentiment_neutral`: Neutral sentiment score (0-1)

### Engagement Metrics
- `likes`: Number of likes/upvotes
- `shares`: Number of shares/retweets
- `comments`: Number of comments/replies
- `views`: Number of views (where applicable)
- `total_engagement`: Sum of likes, shares, and comments
- `engagement_rate_per_1k_followers`: Engagement rate normalized per 1000 followers

### Virality Indicators
- `hours_since_post`: Hours elapsed since posting
- `viral_coefficient`: Engagement velocity (engagement per hour)
- `cross_platform_spread`: Binary indicator for cross-platform viral content

### Additional Metrics
- `toxicity_score`: Content toxicity/controversy score (0-100)
- `location`: Geographic region of user

## 📈 Key Statistics

### Platform Distribution
- **Twitter**: 44,676 posts (29.8%)
- **Reddit**: 37,585 posts (25.1%)
- **Instagram**: 30,195 posts (20.1%)
- **YouTube**: 18,130 posts (12.1%)
- **TikTok**: 11,887 posts (7.9%)
- **Facebook**: 7,527 posts (5.0%)

### Sentiment Distribution
- **Positive**: 67,597 posts (45.1%)
- **Neutral**: 45,110 posts (30.1%)
- **Negative**: 37,293 posts (24.9%)

### Engagement Summary
- **Total Likes**: 53.2 million
- **Total Shares**: 7.9 million
- **Total Comments**: 4.5 million
- **Total Views**: 1.47 billion
- **Verified Users**: 1,806 (1.2%)
- **Cross-Platform Viral Posts**: 1,946

### Geographic Coverage
- North America (34.9%), Asia (25.2%), Europe (25.0%), and more

## 🔬 Methodology

This dataset was created using realistic patterns observed in social media research:
- Engagement metrics follow log-normal distributions typical of social media
- Sentiment scores are generated using patterns from NLP sentiment analysis
- Temporal patterns reflect real posting behaviors (more activity during business hours)
- Platform-specific characteristics (e.g., Twitter's character limit, YouTube's higher view counts)
- Virality patterns based on engagement velocity research

## 💡 Sample Analysis Ideas

1. **Sentiment Trends**: Track how sentiment changes over time for specific topics
2. **Virality Prediction**: Build ML models to predict viral content using engagement patterns
3. **Platform Optimization**: Identify best platforms for different content types
4. **Temporal Analysis**: Discover optimal posting times and day-of-week patterns
5. **Influence vs Engagement**: Analyze relationship between follower count and engagement rate
6. **Cross-Platform Dynamics**: Study how content spreads across platforms
7. **Topic Modeling**: Use NLP to identify emerging trends and topics
8. **Language Comparison**: Compare sentiment patterns across languages
9. **Toxicity Detection**: Analyze relationship between toxicity and engagement
10. **Network Analysis**: Map user influence and content propagation networks

## 🛠️ Recommended Tools & Libraries

- **Python**: pandas, numpy, matplotlib, seaborn, plotly
- **NLP**: nltk, spacy, transformers, textblob
- **Machine Learning**: scikit-learn, xgboost, pytorch
- **Time Series**: statsmodels, prophet
- **Network Analysis**: networkx, igraph
- **Visualization**: plotly, altair, bokeh

## 📊 Sample Queries

### Most Engaging Platform
```python
df.groupby('platform')['total_engagement'].mean().sort_values(ascending=False)
```

### Sentiment by Topic
```python
df.groupby(['topic', 'sentiment_category']).size().unstack()
```

### Viral Content Characteristics
```python
viral_posts = df[df['viral_coefficient'] > 100]
viral_posts.groupby('platform')['viral_coefficient'].describe()
```

### Best Posting Times
```python
df.groupby('hour_of_day')['engagement_rate_per_1k_followers'].mean().plot()
```

## 📚 Citation

If you use this dataset in your research or analysis, please cite:

```
Multi-Platform Social Sentiment Evolution Dataset
Created: October 2025
Platforms: Twitter/X, Reddit, Instagram, YouTube, TikTok, Facebook
Source: Kaggle Datasets
```

## 📄 License

This dataset is released under **CC0: Public Domain**. You are free to use, modify, and distribute this data for any purpose.

## ⚠️ Important Notes

- All user IDs are anonymized for privacy
- Post IDs are synthetic and do not correspond to actual social media posts
- Sentiment scores are based on statistical patterns, not actual content analysis
- This dataset is designed for educational and research purposes

## 🤝 Acknowledgments

This dataset was created to support research in social media analytics, sentiment analysis, and viral content prediction.

## 📧 Contact

For questions or feedback about this dataset, please use the Kaggle discussion forum.

---

**Version**: 1.0  
**Last Updated**: October 2025  
**Data Quality**: 100% complete, 0% missing values  
**Unique Posts**: 150,000  
**Unique Users**: 129,556
