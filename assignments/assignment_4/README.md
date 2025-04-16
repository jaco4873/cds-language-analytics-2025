# Topic Analysis of News Headlines with BERTopic

This project analyzes news headlines using BERTopic to discover underlying topics and how they relate to the predefined news categories.

## Overview

News headlines are concise, information-dense snippets that often contain the core message of a news article. By applying topic modeling to headlines, we can uncover common themes across news stories and identify patterns in how headlines are crafted for different news categories.

This project:
1. Uses pretrained BERT embeddings to represent news headlines
2. Applies BERTopic to cluster these embeddings into coherent topics
3. Analyzes the relationship between discovered topics and predefined news categories
4. Visualizes these relationships to gain insights into news reporting patterns

## Project Structure

```
assignment_4/
├── data/                           # Data directory
│   ├── News_Category_Dataset_v3_subset.jsonl
│   └── embeddings_headlines.npy
├── output/                         # Output visualizations
├── src/                            # Source code
│   ├── config/                     # Configuration
│   │   └── settings.py             # Model and visualization settings
│   ├── data/                       # Data handling
│   │   └── loader.py               # Data loading utilities
│   ├── models/                     # Model definitions
│   │   └── topic_model.py          # BERTopic model wrapper
│   ├── visualization/              # Visualization tools
│   │   └── plotter.py              # Plotting functions
│   └── main.py                     # Main execution script
├── README.md                       # This file
├── pyproject.toml                  # Project dependencies
├── run.sh                          # Execution script
└── setup.sh                        # Environment setup script
```

## Getting Started

### Quick Start (Recommended)

Simply run the analysis script:

```bash
chmod +x run.sh
./run.sh
```

That's it! The script will:
1. Check if the environment is properly set up
2. If not, it will offer to run the setup script for you
3. Run the topic modeling analysis
4. Generate visualizations in the `output` directory

No manual environment setup needed - everything is handled automatically.

### Manual Setup (Alternative)

If you prefer to set up the environment separately:

```bash
chmod +x setup.sh
./setup.sh
```

Then run the analysis:

```bash
./run.sh
```

## Analysis Results and Interpretation

### Discovered Topics

Our analysis discovered 65+ distinct topics in the news headlines dataset, with the most significant being:

1. **Outlier/Miscellaneous** (Topic -1): Contains 5469 documents (39% of the dataset) with varied themes including Christmas, Martha Stewart, and miscellaneous content that didn't fit well into other topics
2. **Crime & Police News** (Topic 0): Focus on cops, shootings, and murder (1841 documents)
3. **Food & Recipes** (Topic 1): Content about recipes, cooking, food ingredients (1503 documents)
4. **Football/Super Bowl** (Topic 2): Coverage of Super Bowl, touchdowns, and NFL figures like Manning (255 documents)
5. **Travel & Destinations** (Topic 3): Content about traveling, destinations, and trip planning (246 documents)
6. **Home & Living** (Topic 4): Headlines about tiny houses, home tours, and interior design (234 documents)

The model also identified numerous other topics with fewer documents, including those related to:
- Sports subcategories (NBA basketball, Olympics, soccer/FIFA, baseball, UFC, tennis)
- Comedy and entertainment (various comedians and shows)
- Business and finance (marketing, banks, jobs, taxes)
- Home-related topics (furniture, gardening, cleaning, DIY)
- Travel specifics (hotels, airlines, cruises, beaches, islands)
- Food and drink subcategories (cocktails, wine, coffee)
- Seasonal content (Halloween, St. Patrick's Day, Valentine's Day)

### The Significance of Topic -1

The large proportion of documents (39%) assigned to Topic -1 is significant. In BERTopic, Topic -1 represents an "outlier" category for documents that don't fit well into the more coherent topics. 

This might be a result of one or more of the following ideas:

1. Many news headlines use language patterns that don't cluster easily with others
2. The news dataset contains substantial thematic diversity beyond the main identified topics
3. The granularity of our topic model might not capture all the nuanced themes present in the dataset

### Topic-Category Relationship

Despite the large outlier category, the analysis reveals several interesting patterns in how the discovered topics align with the predefined news categories:

1. **Clear Category Alignment**: Many topics show clear alignment with specific categories. For example:
   - Topic 0 associated with words such as police, shooting, cops and murder predominantly appears in the CRIME category
   - Topic 1 associated with words such as recipes, food and chocolate is heavily concentrated in the FOOD & DRINK category
   - Topic 2 associated with words such bowl, super, touchdown and manning (Football/Super Bowl) is almost exclusively found in the SPORTS category

2. **Highly Specialized Topics**: The model identified many highly specialized topics with fewer documents (25-50 each), such as:
   - Topic 39 (coffee_starbucks_caffeine_espresso)
   - Topic 36 (oil_climate_exxon_fossil)
   - Topic 56 (halloween_costumes_scariest_decorations)
   - Topic 49 (garden_plants_outdoor_gardening)

### Specific Category Insights

When examining specific categories:

- **SPORTS**: Contains multiple specialized sports topics including football (Topic 2), basketball (Topic 5), Olympics (Topic 6), soccer/FIFA (Topic 11), baseball (Topic 15), golf (Topic 28), UFC/combat sports (Topic 33), and tennis (Topic 50). This fine-grained topical structure reveals how sports news is highly segmented by specific sports types.

- **FOOD & DRINK**: Dominated by recipes (Topic 1), but also includes specialized topics like cocktails/beer (Topic 18), wine (Topic 30), and coffee (Topic 39). This reveals distinct subcommunities within food and beverage content.

- **CRIME**: Primarily associated with Topic 0 (police/shooting/murder), but also includes more specific crime-related topics like marijuana/drugs (Topic 53).

- **TRAVEL**: Shows a diverse set of travel subcategories: general travel (Topic 3), airlines (Topic 7), specific destinations like Paris/Italy (Topic 22), islands/Hawaii (Topic 32), cruises (Topic 43), Mexico (Topic 42), Morocco/Africa (Topic 45), and beaches (Topic 64).

- **HOME & LIVING**: Contains distinct home-related topics including tiny houses (Topic 4), furniture/decorating (Topic 26), kitchen/appliances (Topic 46), gardening (Topic 49), and DIY (Topic 51).

- **BUSINESS**: Includes various business subtopics such as leadership (Topic 10), marketing/startups (Topic 12), banking/mortgage (Topic 14), jobs/Wall Street (Topic 23), women in business (Topic 24), oil/energy (Topic 36), and tax/IRS (Topic 63). Some unrelated topics like Food & Recipes (Topic 1) and Crime (Topic 0) also appear here with a smaller presence, suggesting occasional cross-category overlap

- **COMEDY**: The COMEDY category is heavily populated by outlier headlines, but several prominent topics such as Colbert (Topic 8), SNL (Topic 17), Trevor Noah (Topic 29), Kimmel (Topic 31), Fallon (Topic 38), Seth Meyers (Topic 41), and Samantha Bee (Topic 48) reveal how specific comedians and shows dominate subclusters.

### Topic Distribution and Category Relationship

The model reveals a hierarchical organization of news content:

1. **Primary Topics**: Large topics (0 and 1) with 1500+ documents each represent major content areas (crime and food)
2. **Secondary Topics**: Mid-sized topics (2-10) with 150-250 documents each covering important but more specific domains
3. **Specialized Topics**: Smaller topics (11-65) with 25-150 documents each representing highly specialized content areas
4. **Outlier Content**: Topic -1 with diverse content that doesn't fit other patterns

### Cultural Data Science Perspective

1. **Topical Granularity**: The discovery of numerous small, specialized topics (like specific sports, destinations, or comedians) reveals how media content uses distinct vocabulary patterns within broader categories, creating specialized "content ecosystems" with their own linguistic markers.

2. **Topic-Category Alignment and Divergence**: While many topics align cleanly with categories (sports topics in SPORTS), others cross category boundaries. 

3. **Content Volume Distribution**: The long-tail distribution of topics (few large topics, many small ones) reflects how news media balances mass-appeal content with niche interests. The largest non-outlier topics (crime, food) represent universally engaging subjects with broad appeal.

4. **Semantic Coherence**: The clear keyword patterns in most topics (e.g., police/shooting/cops or recipes/food/chocolate) demonstrate how news media uses consistent vocabulary clusters within content domains, creating recognizable linguistic patterns that help readers quickly identify content types.

## Visualization Outputs

The analysis generates several visualization types:

1. **Interactive Visualizations** (.html files):
   - Topic word clouds showing key terms for each topic
   - Topic hierarchy showing relationships between topics
   - Topic bar charts showing document distribution across topics
   - Topics per class plots showing topic representation across categories

2. **Static Visualizations** (.png files):
   - Heatmap showing topic distribution across categories
   - Category-specific plots showing top topics for each category
   - Topic-specific plots showing category distribution for top topics

3. **Text Summary**:
   - Analysis summary file with basic statistics about topics and documents

## Configuration

You can adjust model parameters in `src/config/settings.py`:

```python
# Model Configuration
MIN_TOPIC_SIZE: int = 25  # Minimum size of a topic (smaller values create more topics)
REMOVE_STOPWORDS: bool = True  # Remove stop words in representation 
REDUCE_FREQUENT_WORDS: bool = True  # Reduce impact of frequent words
```

Both stopword removal approaches (standard stopwords via CountVectorizer and frequent word reduction via ClassTfidfTransformer) are used in this implementation, which helps produce cleaner topic representations by:
1. Removing standard English stopwords like "the", "and", "of", etc.
2. Reducing the impact of corpus-specific high-frequency terms that might not be traditional stopwords


## Acknowledgments

- This project uses the [BERTopic](https://github.com/MaartenGr/BERTopic) library by Maarten Grootendorst
- The news dataset is derived from the [News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset) on Kaggle
