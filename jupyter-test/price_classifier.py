import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class PriceCategory(Enum):
    EXCELLENT = "Excellent"
    VERY_GOOD = "Very Good"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"
    VERY_POOR = "Very Poor"

@dataclass
class PriceAnalysis:
    category: PriceCategory
    confidence: float
    trend: str
    historical_avg: float
    price_range: Tuple[float, float]

class PriceClassifier:
    def __init__(self, historical_prices: List[float] = None):
        self.historical_prices = historical_prices or []
        self.price_ranges = {
            PriceCategory.EXCELLENT: (0, 5),
            PriceCategory.VERY_GOOD: (5, 8),
            PriceCategory.GOOD: (8, 10),
            PriceCategory.FAIR: (10, 12),
            PriceCategory.POOR: (12, 15),
            PriceCategory.VERY_POOR: (15, float('inf'))
        }
    
    def classify_price(self, price: float) -> PriceAnalysis:
        """Classify a price and provide detailed analysis."""
        # Determine category
        category = next(
            (cat for cat, (low, high) in self.price_ranges.items() 
             if low <= price < high),
            PriceCategory.VERY_POOR
        )
        
        # Calculate confidence based on distance from category boundaries
        low, high = self.price_ranges[category]
        confidence = 1 - min(abs(price - low), abs(price - high)) / (high - low)
        
        # Analyze trend if historical data is available
        trend = "Unknown"
        historical_avg = None
        if self.historical_prices:
            historical_avg = np.mean(self.historical_prices)
            if price > historical_avg:
                trend = "Increasing"
            elif price < historical_avg:
                trend = "Decreasing"
            else:
                trend = "Stable"
        
        return PriceAnalysis(
            category=category,
            confidence=confidence,
            trend=trend,
            historical_avg=historical_avg,
            price_range=(low, high)
        )
    
    def plot_price_distribution(self, prices: List[float] = None):
        """Visualize price distribution and categories."""
        prices = prices or self.historical_prices
        if not prices:
            print("No price data available for visualization")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot price distribution
        plt.hist(prices, bins=20, alpha=0.7, color='blue')
        
        # Add category boundaries
        for category, (low, high) in self.price_ranges.items():
            plt.axvline(x=low, color='red', linestyle='--', alpha=0.5)
            plt.text(low, plt.ylim()[1], category.value, rotation=90, va='top')
        
        plt.title('Price Distribution and Categories')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create sample historical prices
    historical_prices = np.random.normal(10, 3, 100)
    
    # Initialize classifier
    classifier = PriceClassifier(historical_prices)
    
    # Test price classification
    test_prices = [4, 7, 9, 11, 13, 16]
    for price in test_prices:
        analysis = classifier.classify_price(price)
        print(f"\nPrice: {price}")
        print(f"Category: {analysis.category.value}")
        print(f"Confidence: {analysis.confidence:.2%}")
        print(f"Trend: {analysis.trend}")
        if analysis.historical_avg:
            print(f"Historical Average: {analysis.historical_avg:.2f}")
    
    # Visualize price distribution
    classifier.plot_price_distribution() 