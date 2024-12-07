import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mplPath

def plot_points(points):
    """
    Plot a list of 2D points on a graph with fixed view from (-20, -20) to (20, 20),
    and include a blue square from (-10, -10) to (10, 10) above grid lines.
    
    :param points: List of (x,y) coordinate tuples
    """
    # Create the figure and axis
    plt.figure(figsize=(10, 10))
    
    # Set the axis limits and ensure they remain fixed
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    
    # Create grid lines every 2 units
    plt.xticks(range(-20, 21, 2))
    plt.yticks(range(-20, 21, 2))
    
    # Add major grid lines
    plt.grid(which='major', color='gray', linestyle='-', linewidth=0.5, zorder=1)
    
    # Highlight x and y axes
    plt.axhline(y=0, color='k', linewidth=1.5, zorder=2)
    plt.axvline(x=0, color='k', linewidth=1.5, zorder=2)
    
    # Create blue square with high zorder to place it above grid lines
    square = patches.Rectangle(
        (-10, -10),  # Lower-left corner
        20,          # Width (from -10 to 10)
        20,          # Height (from -10 to 10)
        facecolor='green',  # Fill color
        alpha=0.2,   # Transparency (0.0 to 1.0)
        edgecolor='green',  # Border color
        linewidth=2,  # Line thickness
        zorder=3     # Ensure square is drawn above grid lines
    )
    
    # Add the square to the plot
    plt.gca().add_patch(square)
    
    # Create a Path object for the square to check point containment
    square_vertices = [(-10, -10), (-10, 10), (10, 10), (10, -10)]
    square_path = mplPath.Path(square_vertices)
    
    # Separate points and color them
    if points:
        inside_points = []
        outside_points = []
        
        # Categorize points
        for point in points:
            if square_path.contains_point(point):
                inside_points.append(point)
            else:
                outside_points.append(point)
        
        # Plot inside points (black)
        if inside_points:
            x_inside, y_inside = zip(*inside_points)
            plt.scatter(x_inside, y_inside, color='green', marker='o', zorder=4)
        
        # Plot outside points (red)
        if outside_points:
            x_outside, y_outside = zip(*outside_points)
            plt.scatter(x_outside, y_outside, color='red', marker='x', zorder=4)
    
    # Labeling
    plt.title('2D Point Visualization with Blue Square')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    
    # Ensure equal aspect ratio and fixed view
    plt.axis('equal')
    
    # Force the plot to maintain the entire (-20, -20) to (20, 20) view
    plt.gca().set_xlim(-20, 20)
    plt.gca().set_ylim(-20, 20)
    
    # Show the plot
    plt.show()

# Example usage with various point sets
sample_points_1 = [
    (3, 4),
    (-5, 6),
    (10, -7),
    (-2, -3),
    (0, 0)
]

sample_points_2 = [
    (50, 50),  # Points outside original view
    (-50, -50)
]

sample_points_3 = []  # Empty list

# Demonstrate with different point sets
plot_points(sample_points_1)
plot_points(sample_points_2)
plot_points(sample_points_3)