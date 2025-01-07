import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime

class CellAnalyzer:
    def __init__(self, output_dir=None):  # Fixed constructor method name
        """ Initialize the cell analyzer with output directory """
        self.output_dir = Path(output_dir if output_dir else "cell_analysis_results")
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def analyze_cell_contours(self, image_path):
        """ Analyze individual cell contours and calculate red/green area ratios. """
        original = cv2.imread(str(image_path))
        if original is None:
            print(f"Failed to read image: {image_path}")
            return [], None
        img = original.copy()
        
        # Split channels
        b, g, r = cv2.split(img)
        
        # Create masks
        green_mask = cv2.threshold(g, 50, 255, cv2.THRESH_BINARY)[1]
        red_mask = cv2.threshold(r, 50, 255, cv2.THRESH_BINARY)[1]
        
        # Find contours
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cell_analyses = []
        vis_img = original.copy()

        for idx, green_contour in enumerate(green_contours):
            green_area = cv2.contourArea(green_contour)
            associated_red_contours = []
            red_area = 0
            
            for red_contour in red_contours:
                M = cv2.moments(red_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    if cv2.pointPolygonTest(green_contour, (cx, cy), False) >= 0:
                        red_area += cv2.contourArea(red_contour)
                        associated_red_contours.append(red_contour)

            if green_area > 0:
                ratio = red_area / green_area
                cell_analyses.append({
                    'cell_index': idx,
                    'green_area': green_area,
                    'red_area': red_area,
                    'ratio': ratio,
                    'circularity': 4 * np.pi * green_area / (cv2.arcLength(green_contour, True) ** 2),
                    'perimeter': cv2.arcLength(green_contour, True)
                })

            # Draw contours on visualization image
            cv2.drawContours(vis_img, [green_contour], -1, (0, 255, 0), 2)
            for red_contour in associated_red_contours:
                cv2.drawContours(vis_img, [red_contour], -1, (0, 0, 255), 2)

            # Add cell index label
            M = cv2.moments(green_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(vis_img, f'#{idx}', (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return cell_analyses, vis_img

    def process_batch_images(self, folder_path):
    
        folder = Path(folder_path)
        all_results = []

        # Create visualization directory
        vis_dir = self.output_dir / f"visualizations_{self.timestamp}"
        vis_dir.mkdir(exist_ok=True)

        # Support multiple image extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(folder.glob(ext))

        if not image_files:
            print(f"No images found in {folder}. Check if the path is correct and images exist.")
            return []

        for image_path in image_files:
            print(f"Processing {image_path.name}...")
            results, vis_img = self.analyze_cell_contours(image_path)
            if vis_img is not None:
                # Save visualization
                vis_path = vis_dir / f"vis_{image_path.stem}.png"
                cv2.imwrite(str(vis_path), vis_img)
            all_results.append({
                'image': image_path.name,
                'cells': results
            })

        return all_results


    def generate_statistics(self, results):
        """ Generate comprehensive statistical analysis """
        all_data = []
        
        for image_result in results:
            for cell in image_result['cells']:
                cell_data = {
                    'image': image_result['image'],
                    'cell_index': cell['cell_index'],
                    'ratio': cell['ratio'],
                    'green_area': cell['green_area'],
                    'red_area': cell['red_area'],
                    'circularity': cell['circularity'],
                    'perimeter': cell['perimeter']
                }
                all_data.append(cell_data)

        if not all_data:
            print("No data to analyze!")
            return {}, pd.DataFrame()

        df = pd.DataFrame(all_data)

        stats_dict = {
            'ratio': df['ratio'].describe(),
            'green_area': df['green_area'].describe(),
            'red_area': df['red_area'].describe(),
            'circularity': df['circularity'].describe(),
            'perimeter': df['perimeter'].describe()
        }

        stats_dict['ratio_normality'] = stats.normaltest(df['ratio'])
        stats_dict['area_correlation'] = stats.pearsonr(df['green_area'], df['red_area'])

        return stats_dict, df

    def create_visualizations(self, df):
        """ Create statistical visualizations """
        if df.empty:
            print("No data to visualize!")
            return

        fig_dir = self.output_dir / f"figures_{self.timestamp}"
        fig_dir.mkdir(exist_ok=True)

        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='ratio', bins=30)
        plt.title('Distribution of Red/Green Area Ratios')
        plt.savefig(fig_dir / 'ratio_distribution.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='green_area', y='red_area')
        plt.title('Correlation between Green and Red Areas')
        plt.savefig(fig_dir / 'area_correlation.png')
        plt.close()

        plt.figure(figsize=(12, 6))
        df.boxplot(column=['ratio', 'circularity'])
        plt.title('Box Plots of Key Metrics')
        plt.savefig(fig_dir / 'metrics_boxplot.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='circularity', y='ratio')
        plt.title('Circularity vs Ratio')
        plt.savefig(fig_dir / 'circularity_vs_ratio.png')
        plt.close()

    def run_analysis(self, input_folder):
        """ Run the complete analysis pipeline """
        print(f"Starting analysis of {input_folder}")
        print("This may take a few minutes depending on the number of images...")

        # Process images
        results = self.process_batch_images(input_folder)

        if not results:
            print("No results to analyze!")
            return [], {}, pd.DataFrame()  # Return empty structures instead of None

        # Generate statistics
        stats_dict, df = self.generate_statistics(results)

        # Create visualizations
        self.create_visualizations(df)

        # Export results to CSV
        csv_path = self.output_dir / f"cell_analysis_{self.timestamp}.csv"
        df.to_csv(csv_path, index=False)

        # Export statistics to text file
        stats_path = self.output_dir / f"statistics_{self.timestamp}.txt"
        with open(stats_path, 'w') as f:
            f.write("=== Cell Analysis Statistics ===\n\n")
            for metric, stat in stats_dict.items():
                f.write(f"\n{metric.upper()}\n")
                f.write(str(stat))
                f.write("\n" + "=" * 50 + "\n")

        print("\nAnalysis complete! Check the output directory for:")
        print("1. Visualizations of analyzed cells")
        print("2. Statistical plots and figures")
        print("3. CSV file with all measurements")
        print("4. Detailed statistics report")

        return results, stats_dict, df  # Ensure valid returns here



        
# Create a new file called run_analysis.py with this code:
if __name__ == "__main__":  # Fixed main guard from "_main_" to "__main__"
    input_folder = r"D:\Capstone\Cancer\IMG\carcinoma_M"  # Replace with your input folder path
    output_dir = r"D:\Capstone\Cancer\IMG\car"  # Replace with your desired output directory
    
    analyzer = CellAnalyzer(output_dir)
    results, stats, df = analyzer.run_analysis(input_folder)
