# Multi-Platform Social Sentiment Evolution - Column Descriptions

## Post Identification Columns

**post_id** (String)
- Unique identifier for each post
- Format: PLATFORM_YYYYMMDD_SEQUENCE (e.g., TWI20250419000001)
- Platform prefixes: TWI=Twitter, RED=Reddit, INS=Instagram, YOU=YouTube, TIK=TikTok, FAC=Facebook
- Primary key for the dataset

**platform** (Categorical String)
- Social media platform where content was posted
- Values: Twitter, Reddit, Instagram, YouTube, TikTok, Facebook
- 6 unique platforms

**timestamp** (Datetime String, YYYY-MM-DD HH:MM:SS)
- Exact date and time of post
- Format: 2025-04-19 14:32:15
- Second-level precision

**date** (Date String, YYYY-MM-DD)
- Date of post (without time)
- Format: 2025-04-19
- Used for daily aggregations

## Temporal Feature Columns

**hour_of_day** (Integer)
- Hour when post was created (24-hour format)
- Range: 0-23
- Used to analyze optimal posting times

**day_of_week** (Integer)
- Day of the week (0 = Monday, 6 = Sunday)
- Range: 0-6
- Used to identify weekly patterns

**is_weekend** (Binary: 0 or 1)
- Indicates if post was made on weekend
- 1 = Saturday or Sunday, 0 = Weekday
- Used for weekday vs weekend analysis

## User Metrics Columns

**user_id** (String)
- Anonymized user identifier
- Format: user_XXXXXX (6-digit number)
- 129,556 unique users in dataset

**followers** (Integer)
- Number of followers/subscribers
- Range: varies by platform (1 to 100,000+)
- Log-normal distribution typical of social media

**account_age_days** (Integer)
- Age of the account in days
- Range: 30-3,650 days (minimum 1 month to 10 years)
- Used to assess account credibility

**verified** (Binary: 0 or 1)
- Indicates if account is verified/authenticated
- 1 = Verified, 0 = Not verified
- Only 1.2% of posts from verified accounts

## Content Characteristic Columns

**topic** (Categorical String)
- Primary topic/category of the post
- 15 categories: Technology, Politics, Entertainment, Sports, Health, Climate, Business, Education, Science, Gaming, Food, Travel, Fashion, Finance, AI/ML
- Evenly distributed across topics

**language** (Categorical String)
- Language of the content
- 10 languages: English (dominant at 55%), Spanish, Portuguese, French, German, Japanese, Korean, Hindi, Arabic, Chinese
- Based on global social media usage patterns

**content_length** (Integer)
- Character count of the post content
- Range: 10-500 characters (varies by platform)
- Twitter: 10-280, Reddit: 20-500, Instagram: 10-150

**media_type** (Categorical String)
- Type of media included in post
- Values: Text, Image, Video, Link, Poll
- Distribution: Text (35%), Image (30%), Video (20%), Link (10%), Poll (5%)

**num_hashtags** (Integer)
- Number of hashtags used in the post
- Range: 0-10+
- Follows Poisson distribution (mean ~2)

## Sentiment Analysis Columns

**sentiment_category** (Categorical String)
- Overall sentiment classification
- Values: Positive, Negative, Neutral
- Distribution: Positive (45.1%), Neutral (30.1%), Negative (24.9%)

**sentiment_positive** (Float)
- Positive sentiment probability score
- Range: 0.000-1.000 (3 decimal places)
- Sum of positive + negative + neutral ≈ 1.0

**sentiment_negative** (Float)
- Negative sentiment probability score
- Range: 0.000-1.000 (3 decimal places)
- Higher values indicate more negative sentiment

**sentiment_neutral** (Float)
- Neutral sentiment probability score
- Range: 0.000-1.000 (3 decimal places)
- Ambiguous or balanced sentiment

## Engagement Metrics Columns

**likes** (Integer)
- Number of likes/upvotes/reactions
- Range: 0 to 10,000+ (varies by platform)
- Primary engagement metric

**shares** (Integer)
- Number of shares/retweets/reposts
- Range: 0 to 2,500+
- Typically 5-25% of likes

**comments** (Integer)
- Number of comments/replies
- Range: 0 to 1,500+
- Typically 2-15% of likes

**views** (Integer)
- Number of views (where applicable)
- Range: 0 to 500,000+
- Particularly high for video content (YouTube, TikTok)

**total_engagement** (Integer)
- Sum of likes + shares + comments
- Aggregate engagement metric
- Used for overall popularity assessment

**engagement_rate_per_1k_followers** (Float)
- Normalized engagement rate
- Formula: (total_engagement / followers) * 1000
- Allows comparison across different follower counts

## Virality Indicator Columns

**hours_since_post** (Float)
- Hours elapsed since posting
- Range: 0.1-48.0 hours
- Used to calculate engagement velocity

**viral_coefficient** (Float)
- Engagement velocity metric
- Formula: total_engagement / hours_since_post
- Higher values indicate faster spreading content

**cross_platform_spread** (Binary: 0 or 1)
- Indicates if content went viral across multiple platforms
- 1 = Cross-platform viral, 0 = Single platform
- 1,946 posts (1.3%) achieved cross-platform spread

## Additional Metrics Columns

**toxicity_score** (Float)
- Content toxicity/controversy score
- Range: 0.0-100.0
- Higher scores indicate more toxic/controversial content
- Correlated with negative sentiment

**location** (Categorical String)
- Geographic region of the user
- 7 regions: North America, Europe, Asia, South America, Africa, Oceania, Unknown
- Distribution: North America (34.9%), Asia (25.2%), Europe (25.0%)

---

## Data Types Summary

- **String**: post_id, platform, timestamp, date, user_id, topic, language, media_type, sentiment_category, location
- **Integer**: hour_of_day, day_of_week, is_weekend, followers, account_age_days, verified, content_length, num_hashtags, likes, shares, comments, views, total_engagement, cross_platform_spread
- **Float**: sentiment_positive, sentiment_negative, sentiment_neutral, engagement_rate_per_1k_followers, hours_since_post, viral_coefficient, toxicity_score

## Missing Values

**All columns**: 0% missing (100% complete)

## Key Relationships

- **Engagement vs Followers**: Log-normal relationship
- **Sentiment vs Engagement**: Positive content tends to have higher engagement
- **Time of Day vs Engagement**: Peak engagement during afternoon/evening hours
- **Verified vs Engagement**: Verified accounts have ~30% higher engagement
- **Platform vs Metrics**: Each platform has characteristic engagement patterns

## Platform-Specific Notes

- **Twitter**: Character limit 280, high share rate
- **Reddit**: Karma-based system, text-heavy
- **Instagram**: Visual-first, high image/video ratio
- **YouTube**: Video content, very high view counts
- **TikTok**: Short-form video, high virality potential
- **Facebook**: Mixed content, moderate engagement
