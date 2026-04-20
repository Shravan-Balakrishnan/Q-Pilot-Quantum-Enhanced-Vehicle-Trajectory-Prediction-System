"""
Research module for analyzing academic papers related to quantum machine learning
and vehicle trajectory prediction.
"""
import PyPDF2
import re
import nltk
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class PaperAnalyzer:
    """
    Analyzer for research papers
    """
    def __init__(self):
        """
        Initialize paper analyzer
        """
        # Download required NLTK data (would be done once)
        # nltk.download('punkt')
        # nltk.download('stopwords')
        pass

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF file

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            str: Extracted text
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def extract_abstract(self, text):
        """
        Extract abstract from paper text

        Args:
            text (str): Full paper text

        Returns:
            str: Extracted abstract
        """
        # Look for common abstract patterns
        abstract_patterns = [
            r'abstract\s*\n\s*(.*?)\n\s*\n',
            r'abstract\s*:\s*(.*?)\n\n',
            r'introduction\s*\n',
            r'index terms\s*\n'
        ]

        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1) if match.groups() else ""
                # Limit length for practicality
                return abstract[:1000] + "..." if len(abstract) > 1000 else abstract

        # Fallback: first 300 words
        words = text.split()[:300]
        return " ".join(words) + "..."

    def extract_keywords(self, text, top_n=10):
        """
        Extract keywords from text

        Args:
            text (str): Text to analyze
            top_n (int): Number of top keywords to return

        Returns:
            list: Top keywords
        """
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

        # Split into words
        words = text.split()

        # Remove common stopwords (simplified list)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }

        # Filter words
        filtered_words = [word for word in words if word not in stopwords and len(word) > 2]

        # Count frequencies
        word_freq = Counter(filtered_words)

        # Return top keywords
        return [word for word, freq in word_freq.most_common(top_n)]

    def analyze_paper(self, pdf_path):
        """
        Complete analysis of a research paper

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            dict: Analysis results
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)

        if not text:
            return {"error": "Could not extract text from PDF"}

        # Extract components
        abstract = self.extract_abstract(text)
        keywords = self.extract_keywords(text)

        # Basic statistics
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?]+', text))

        return {
            "abstract": abstract,
            "keywords": keywords,
            "statistics": {
                "word_count": word_count,
                "sentence_count": sentence_count
            }
        }

    def compare_papers(self, paper_texts):
        """
        Compare multiple papers using TF-IDF similarity

        Args:
            paper_texts (list): List of paper texts

        Returns:
            np.array: Similarity matrix
        """
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(paper_texts)

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        return similarity_matrix

    def generate_insights(self, papers_data):
        """
        Generate insights from analyzed papers

        Args:
            papers_data (list): List of paper analysis results

        Returns:
            dict: Generated insights
        """
        all_keywords = []
        all_abstracts = []

        for paper in papers_data:
            if "keywords" in paper:
                all_keywords.extend(paper["keywords"])
            if "abstract" in paper:
                all_abstracts.append(paper["abstract"])

        # Most common keywords
        keyword_counter = Counter(all_keywords)
        top_keywords = keyword_counter.most_common(10)

        # Research trends
        trends = self._identify_trends(all_abstracts)

        return {
            "top_keywords": top_keywords,
            "research_trends": trends
        }

    def _identify_trends(self, abstracts):
        """
        Identify research trends from abstracts

        Args:
            abstracts (list): List of abstracts

        Returns:
            list: Identified trends
        """
        # Keywords associated with trends
        trend_keywords = {
            "quantum_advantage": ["advantage", "superior", "better", "outperform"],
            "hybrid_approach": ["hybrid", "combined", "integration", "classical"],
            "scalability": ["scale", "large", "practical", "real-world"],
            "applications": ["autonomous", "vehicle", "traffic", "navigation"]
        }

        trends = []

        full_text = " ".join(abstracts).lower()

        for trend, keywords in trend_keywords.items():
            count = sum(1 for keyword in keywords if keyword in full_text)
            if count > 0:
                trends.append(trend)

        return trends


class ResearchDatabase:
    """
    Database for managing research papers and findings
    """
    def __init__(self, db_path="../data/research_papers.csv"):
        """
        Initialize research database

        Args:
            db_path (str): Path to CSV database file
        """
        self.db_path = db_path
        self.papers_df = self._load_database()

    def _load_database(self):
        """
        Load paper database

        Returns:
            pd.DataFrame: Papers database
        """
        try:
            return pd.read_csv(self.db_path)
        except FileNotFoundError:
            # Create empty database
            columns = ['title', 'authors', 'year', 'venue', 'abstract', 'keywords', 'summary']
            return pd.DataFrame(columns=columns)

    def add_paper(self, paper_data):
        """
        Add paper to database

        Args:
            paper_data (dict): Paper information
        """
        new_row = pd.DataFrame([paper_data])
        self.papers_df = pd.concat([self.papers_df, new_row], ignore_index=True)
        self._save_database()

    def _save_database(self):
        """
        Save database to file
        """
        self.papers_df.to_csv(self.db_path, index=False)

    def search_papers(self, query, top_n=5):
        """
        Search papers by query

        Args:
            query (str): Search query
            top_n (int): Number of results to return

        Returns:
            pd.DataFrame: Search results
        """
        # Simple keyword search in title and abstract
        query_lower = query.lower()
        matches = self.papers_df[
            self.papers_df['title'].str.contains(query_lower, case=False, na=False) |
            self.papers_df['abstract'].str.contains(query_lower, case=False, na=False)
        ]

        return matches.head(top_n)

    def get_recent_papers(self, years=5):
        """
        Get recent papers

        Args:
            years (int): Number of years to look back

        Returns:
            pd.DataFrame: Recent papers
        """
        current_year = pd.Timestamp.now().year
        recent_papers = self.papers_df[
            pd.to_numeric(self.papers_df['year'], errors='coerce') >= (current_year - years)
        ]
        return recent_papers


def main():
    """
    Main function for testing the paper analyzer
    """
    print("Research Paper Analyzer module ready for use.")
    print("Use PaperAnalyzer class to analyze research papers.")


if __name__ == "__main__":
    main()