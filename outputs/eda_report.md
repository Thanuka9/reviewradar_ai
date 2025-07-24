# EDA Report for Yelp Dataset
**Run on:** 2025-07-25T00:51:55.021453
**Environment:** Python 3.11.0, pandas 2.3.1, SQLAlchemy 2.0.41, matplotlib 3.10.3, statsmodels 0.14.5
---

## 1) Reviews Sample
- Loaded **100000** reviews (sample)

## 2) Star Rating Distribution
|   stars |   count |
|--------:|--------:|
|       1 |   10917 |
|       2 |    7983 |
|       3 |   11362 |
|       4 |   25333 |
|       5 |   44405 |
![Stars](outputs/star_distribution.png)

## 3) Review Length Distribution
|       |   text_length |
|:------|--------------:|
| count |    100000     |
| mean  |       548.26  |
| std   |       501.673 |
| min   |         3     |
| 25%   |       225     |
| 50%   |       394     |
| 75%   |       692     |
| max   |      5000     |
![Length](outputs/review_length.png)

## 4) Top 20 Business Categories
| categories                |   count |
|:--------------------------|--------:|
| Restaurants               |   52268 |
| Food                      |   27781 |
| Shopping                  |   24395 |
| Home Services             |   14356 |
| Beauty & Spas             |   14292 |
| Nightlife                 |   12281 |
| Health & Medical          |   11890 |
| Local Services            |   11198 |
| Bars                      |   11065 |
| Automotive                |   10773 |
| Event Planning & Services |    9895 |
| Sandwiches                |    8366 |
| American (Traditional)    |    8139 |
| Active Life               |    7687 |
| Pizza                     |    7093 |
| Coffee & Tea              |    6703 |
| Fast Food                 |    6472 |
| Breakfast & Brunch        |    6239 |
| American (New)            |    6097 |
| Hotels & Travel           |    5857 |
![Categories](outputs/top_categories.png)

## 5) Users by Join Year
|   since |   count |
|--------:|--------:|
|    2004 |      90 |
|    2005 |     937 |
|    2006 |    5423 |
|    2007 |   15340 |
|    2008 |   31097 |
|    2009 |   64911 |
|    2010 |  109054 |
|    2011 |  176435 |
|    2012 |  195955 |
|    2013 |  209762 |
|    2014 |  233465 |
|    2015 |  247850 |
|    2016 |  217620 |
|    2017 |  151024 |
|    2018 |  133568 |
|    2019 |  104655 |
|    2020 |   47444 |
|    2021 |   40485 |
|    2022 |    2782 |
![Join Year](outputs/users_by_join_year.png)

## 6) Monthly Review Volume
| month               |   cnt |
|:--------------------|------:|
| 2005-02-01 00:00:00 |     3 |
| 2005-03-01 00:00:00 |    74 |
| 2005-04-01 00:00:00 |    26 |
| 2005-05-01 00:00:00 |   108 |
| 2005-06-01 00:00:00 |    38 |
| 2005-07-01 00:00:00 |   259 |
| 2005-08-01 00:00:00 |    66 |
| 2005-09-01 00:00:00 |    69 |
| 2005-10-01 00:00:00 |    39 |
| 2005-11-01 00:00:00 |    70 |
| 2005-12-01 00:00:00 |   102 |
| 2006-01-01 00:00:00 |   221 |
| 2006-02-01 00:00:00 |   185 |
| 2006-03-01 00:00:00 |   191 |
| 2006-04-01 00:00:00 |   196 |
![Monthly](outputs/monthly_volume.png)

## 7) Average Rating by Category (Top 20)
| cat                              |   stars |
|:---------------------------------|--------:|
| Acne Treatment                   |       5 |
| Yelp Events                      |       5 |
| & Probates                       |       5 |
| Drywall Installation & Repair    |       5 |
| Ear Nose & Throat                |       5 |
| Employment Law                   |       5 |
| Environmental Testing            |       5 |
| Estate Planning Law              |       5 |
| Ethical Grocery                  |       5 |
| Tenant and Eviction Law          |       5 |
| Excavation Services              |       5 |
| Arabic                           |       5 |
| Trailer Dealers                  |       5 |
| Traffic Schools                  |       5 |
| Title Loans                      |       5 |
| Sicilian                         |       5 |
| Auto Upholstery                  |       5 |
| Screen Printing/T-Shirt Printing |       5 |
| Tattoo Removal                   |       5 |
| Art Restoration                  |       5 |
![AvgRating](outputs/avg_rating_by_category.png)

## 8) Reviews per Business Distribution
|       |         cnt |
|:------|------------:|
| count | 150346      |
| mean  |     46.4944 |
| std   |    124.519  |
| min   |      5      |
| 25%   |      8      |
| 50%   |     15      |
| 75%   |     38      |
| max   |   7673      |
![BizDist](outputs/reviews_per_business.png)

## 9) Review Length vs. Star Rating
![LenVsStar](outputs/length_by_star.png)

## 10) Top 20 Most Active Users
| user_id                |   review_count |
|:-----------------------|---------------:|
| Hi10sGSZNxQH3NLyWSZ1oA |          17473 |
| 8k3aO-mPeyhbR5HUucA5aA |          16978 |
| hWDybu_KvYLSdEFzGrniTw |          16567 |
| RtGqdDBvvBCjcu5dUqwfzA |          12868 |
| P5bUL3Engv-2z6kKohB6qQ |           9941 |
| nmdkHL2JKFx55T3nq5VziA |           8363 |
| bQCHF5rn5lMI9c5kEwCaNA |           8354 |
| 8RcEwGrFIgkt9WQ35E6SnQ |           7738 |
| Xwnf20FKuikiHcSpcEbpKQ |           6766 |
| CxDOIDnH8gp9KXzpBHJYXw |           6679 |
| IucvvxdQXXhjQ4z6Or6Nrw |           6459 |
| HFECrzYDpgbS5EmTBtj2zQ |           5887 |
| m07sy7eLtOjVdZ8oN9JKag |           5800 |
| kS1MQHYwIfD0462PE61IBw |           5511 |
| IlGYj_XAMG3v75rfmtBs_Q |           5434 |
| Eypq5gLLjCapBVVnMw_MyA |           5163 |
| U4INQZOPSUaj8hMjLlZ3KA |           5061 |
| bLbSNkLggFnqwNNzzq-Ijw |           5014 |
| wZPizeBxMAyOSl0M0zuCjg |           5002 |
| GHoG4X4FY8D8L563zzPX5w |           4994 |
![TopUsers](outputs/top_users.png)

## 11) Sentiment vs. Star Rating
- Pearson correlation (stars, sentiment) = 0.624
![Sentiment](outputs/sentiment_by_star.png)

## 12) Seasonal Decomposition (if ≥24 months)
- Skipped: only 15 months of data available

## 13) User Retention (Lifetime in Days)
|       |             d |
|:------|--------------:|
| count |    1.9879e+06 |
| mean  |  339.403      |
| std   |  705.382      |
| min   |    0          |
| 25%   |    0          |
| 50%   |    0          |
| 75%   |  295          |
| max   | 6003          |
![Retention](outputs/user_retention.png)
