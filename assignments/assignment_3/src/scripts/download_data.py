"""Script to download and prepare the Gutenberg dataset."""
import click
import nltk
from pathlib import Path
from nltk.corpus import gutenberg
from ..config.settings import settings
from ..utils.logger import logger, log_section

def download_gutenberg() -> list[str]:
    """Download the Gutenberg dataset from NLTK."""
    nltk.download('gutenberg', quiet=False)
    return gutenberg.fileids()

def save_gutenberg_texts(fileids: list[str]):
    """Save each Gutenberg text as a separate file."""
    logger.info("Downloading Gutenberg dataset...")
    
    # Create gutenberg directory if it doesn't exist
    gutenberg_dir = settings.DATA_DIR / "gutenberg"
    gutenberg_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each text file
    for fileid in fileids:
        # Get the raw text
        text = gutenberg.raw(fileid)
        
        # Create filename without the .txt extension to avoid double extension
        filename = Path(fileid).stem + ".txt"
        output_path = gutenberg_dir / filename
        
        # Save the text
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        logger.debug(f"Saved {filename}")

@click.command()
def main():
    """Download and prepare the Gutenberg dataset."""
    log_section("Downloading Gutenberg Dataset", "📚")
    fileids = download_gutenberg()
    
    logger.info(f"Found {len(fileids)} texts:")
    for fileid in fileids:
        logger.info(f"  • {fileid}")
    
    log_section("Saving Texts", "💾")
    save_gutenberg_texts(fileids)
    
    logger.info(f"Texts saved to {settings.DATA_DIR}/gutenberg/")

if __name__ == "__main__":
    main() 