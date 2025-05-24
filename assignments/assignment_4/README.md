# Assignment 4: Topic Analysis of News Headlines with BERTopic

## Introduction 
This project analyzes news headlines using BERTopic to discover underlying topics and how they relate to the predefined news categories. News headlines are concise, information-dense snippets that often contain the core message of a news article. By applying topic modeling to headlines, we can uncover common themes across news stories and identify patterns in how headlines are crafted for different news categories.

## Data
The analysis uses a subset of the News Category Dataset from Kaggle, containing headlines from select news categories. The dataset includes both the headlines and their assigned categories (e.g., SPORTS, FOOD & DRINK, CRIME). To facilitate efficient processing, we utilize pre-computed BERT embeddings for the headlines, stored in the data directory.

## Project Structure

```
assignment_4/
├── data/                           # Data files
├── output/                         # Generated visualizations
├── src/                            # Source code
│   ├── config/                     # Configuration
│   │   └── settings.py             # Model and visualization settings
│   ├── utils/                      # Data handling
│   │   └── data_loader.py          # Data loading utility
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
./setup.sh
```

Then run the analysis:

```bash
./run.sh
```

### Configuration

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

## Methods

The analysis workflow consists of:

1. Data Preparation: Loading news headlines, categories, and pre-computed embeddings
2. Topic Modeling: Applying BERTopic with customized parameters (min_topic_size=25)
3. Topic Analysis: Associating topics with documents and extracting topic information
4. Visualization: Generating various visualizations to explore topic-category relationships, including:
   - Interactive Visualizations (.html files): Topic word clouds showing key terms for each topic, topic hierarchy showing relationships between topics, topic bar charts showing document distribution across topics, and topics per class plots showing topic representation across categories
   - Static Visualizations (.png files): Heatmap showing topic distribution across categories, category-specific plots showing top topics for each category, and topic-specific plots showing category distribution for top topics
   - Text Summary: Analysis summary file with basic statistics about topics and documents

The model configuration uses several techniques to improve topic quality:
- Minimum topic size of 25 documents to focus on meaningful topics
- Stopword removal to eliminate common non-informative words
- Reduction of frequent words' impact to prevent common domain-specific terms from dominating topics

## Results

The analysis discovered 65+ distinct topics in the news headlines, with significant findings including:

### Key Discovered Topics

Our analysis identified several prominent topics:

1. Outlier/Miscellaneous (Topic -1): Contains 5469 documents (39% of the dataset) with varied themes that didn't fit well into other topics
2. Crime & Police News (Topic 0): Focus on cops, shootings, and murder (1841 documents)
3. Food & Recipes (Topic 1): Content about recipes, cooking, food ingredients (1503 documents)
4. Football/Super Bowl (Topic 2): Coverage of Super Bowl, touchdowns, and NFL figures (255 documents)
5. Travel & Destinations (Topic 3): Content about traveling and trip planning (246 documents)
6. Home & Living (Topic 4): Headlines about tiny houses and interior design (234 documents)

The large proportion of documents (39%) assigned to Topic -1 is significant. In BERTopic, Topic -1 represents outlier documents that don't fit well into more coherent topics, suggesting substantial thematic diversity in news headlines that resists simple categorization.

### Topic Distribution

- Primary Topics: Two major topics (Crime & Police News, Food & Recipes) with 1500+ documents each
- Secondary Topics: Several mid-sized topics (250+ documents) covering specific domains like Football, Travel, and Home & Living
- Specialized Topics: Numerous smaller topics (25-150 documents) for niche content areas, including specialized sports topics, comedy shows, business subcategories, and seasonal content

### Topic-Category Alignment

The analysis shows clear alignment between discovered topics and predefined categories:

#### SPORTS
Contains distinct subclusters for different sports:
- Football (Topic 2): Super Bowl, touchdowns, NFL figures
- Basketball (Topic 5): NBA, games, players
- Olympics (Topic 6): Olympic athletes, medals, events
- Soccer (Topic 11): FIFA, World Cup, teams
- Baseball (Topic 15): MLB, games, teams

#### FOOD & DRINK
Dominated by recipes (Topic 1), but includes specialized topics:
- Cocktails/Beer (Topic 18): brewing, mixology, beer types
- Wine (Topic 30): vineyards, wine varieties, tasting
- Coffee (Topic 39): Starbucks, caffeine, espresso

#### CRIME
Primarily associated with police/shooting/murder topics (Topic 0), with additional subtopics like marijuana/drugs (Topic 53)

#### TRAVEL
Shows diverse travel subcategories:
- General travel (Topic 3): destinations, planning
- Airlines (Topic 7): flights, airports, travel tips
- Specific destinations like Paris/Italy (Topic 22)
- Islands/Hawaii (Topic 32)
- Cruises (Topic 43)

#### COMEDY
Organized around specific comedians and shows:
- Colbert (Topic 8)
- SNL (Topic 17)
- Trevor Noah (Topic 29)
- Kimmel (Topic 31)
- Fallon (Topic 38)

#### BUSINESS
Includes various business subtopics:
- Leadership (Topic 10)
- Marketing/startups (Topic 12)
- Banking/mortgage (Topic 14)
- Jobs/Wall Street (Topic 23)
- Oil/energy (Topic 36)
- Tax/IRS (Topic 63)

### Significant Patterns

1. Hierarchical Content Organization: News content follows a clear structure from broad topics to specialized niches
2. Category-Specific Language: Each topic uses distinct vocabulary patterns that signal its content domain
3. Cross-Category Topics: Some topics appear across multiple categories, revealing thematic overlaps in news coverage

## Discussion

From a cultural data science perspective, this analysis offers several insights:

1. Topical Ecosystems: News media creates specialized "content ecosystems" with distinct linguistic patterns, allowing readers to quickly identify content types.

2. Content Volume Distribution: The long-tail distribution of topics (few large topics, many small ones) reflects how media balances mass-appeal content with niche interests. The largest non-outlier topics (crime, food) represent universally engaging subjects.

3. Semantic Coherence: Clear keyword patterns within topics demonstrate how news media uses consistent vocabulary clusters that help readers navigate content efficiently.

4. Category Fluidity: While many topics align cleanly with categories, others cross boundaries, showing how news content often defies strict categorization and reflects the complexity of real-world events.

The high proportion of outlier headlines (39%) suggests significant thematic diversity that may require more granular modeling or indicates the inherently diverse nature of news content that resists simple categorization.
## Acknowledgments

- This project uses the [BERTopic](https://github.com/MaartenGr/BERTopic) library by Maarten Grootendorst
- The news dataset is derived from the [News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset) on Kaggle
