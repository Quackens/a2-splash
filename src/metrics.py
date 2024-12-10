import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mplPath
import matplotlib.animation as animation
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class RealTimePlotter:
    def __init__(self):
        """
        Plot a list of 2D points on a graph with fixed view from (-20, -20) to (20, 20),
        and include a blue square from (-10, -10) to (10, 10) above grid lines.
        
        :param points: List of (x,y) coordinate tuples
        """
        # Create the figure and axis
        self.fig = plt.figure(figsize=(10, 10))

        # Initialize point storage
        self.points = []

        # Set the axis limits and ensure they remain fixed
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        
        # Create grid lines every 2 units
        plt.xticks(range(-20, 21, 2))
        plt.yticks(range(-20, 21, 2))

        # Add major grid lines
        # plt.gca().set_facecolor('#faefb4')  # Set light yellow background color
        plt.grid(which='major', color='gray', linestyle='-', linewidth=0.5, zorder=1)
        
        # Highlight x and y axes
        plt.axhline(y=0, color='k', linewidth=1.5, zorder=2)
        plt.axvline(x=0, color='k', linewidth=1.5, zorder=2)

        # Labeling
        plt.title('Splash Metrics & Visual Feedback')
        plt.xlabel('X-axis')
        plt.ylabel('Z-axis')
        
        # Ensure equal aspect ratio and fixed view
        plt.axis('equal')
        
        # Force the plot to maintain the entire (-20, -20) to (20, 20) view
        self.fig.gca().set_xlim(-20, 20)
        self.fig.gca().set_ylim(-20, 20)

        # Create blue square with high zorder to place it above grid lines
        self.square = patches.Rectangle(
            (-10, -10),  # Lower-left corner
            20,          # Width (from -10 to 10)
            20,          # Height (from -10 to 10)
            facecolor='green',  # Fill color
            alpha=0.2,   # Transparency (0.0 to 1.0)
            edgecolor='green',  # Border color
            linewidth=2,  # Line thickness
            zorder=3     # Ensure square is drawn above grid lines
        )

        self.circle = patches.Circle(
            (0, 0), # Centered at 0, 0
            radius=4.6,  # radius of 4.6cm --> diameter = 9.2cm == 92mm
            facecolor='blue',  # Fill color
            alpha=0.3,   # Transparency (0.0 to 1.0)
            edgecolor='blue',  # Border color
            linewidth=2,  # Line thickness
            zorder=4     # Ensure square is drawn above grid lines
        )

        # Add the square to the plot
        self.fig.gca().add_patch(self.square)
        self.fig.gca().add_patch(self.circle)

        # Create a Path object for the square & circle to check point containment
        self.square_vertices = [(-10, -10), (-10, 10), (10, 10), (10, -10)]
        self.square_path = mplPath.Path(self.square_vertices)
        self.circle_path = mplPath.Path.circle(radius=4.6)

        # Scatter plot for points
        self.scatter = plt.scatter([], [], c=[], cmap='coolwarm')

        # enable plotting points on mouse click
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # Session metrics
        self.total_shots = 0
        self.splash_shots = 0
        self.avg_percentage = 0.0
        self.avg_text = f' Session Accuracy: {self.avg_percentage:.1f}% ({self.splash_shots}/{self.total_shots})'
        self.avg_display = self.fig.text(0.02, 0.98, self.avg_text,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


    def add_point(self, x, y):
        """
        Add a new point to the plot.
        :param x: x-coordinate
        :param y: y-coordinate
        """
        print(f"x={x}, y={y}")
        point = (x, y)
        self.points.append(point)
        self.total_shots += 1
        if self.circle_path.contains_point(point): 
            self.splash_shots += 1
        self.update_plot()

    def update_plot(self):
        """
        Update the scatter plot with current points
        """
        # Clear previous scatter plot
        self.scatter.remove()
        
        if not self.points:
            return
        
        # Separate points by location
        splash_xs = []
        splash_ys = []
        in_xs = []
        in_ys = []
        out_xs = []
        out_ys = []
        
        for point in self.points:
            if self.circle_path.contains_point(point):
                splash_xs.append(point[0])
                splash_ys.append(point[1])
            # Color points based on location
            elif self.square_path.contains_point(point):
                in_xs.append(point[0])
                in_ys.append(point[1])
            else:
                out_xs.append(point[0])
                out_ys.append(point[1])

        avg_percentage = (self.splash_shots / self.total_shots) * 100 if self.total_shots > 0 else 0
        avg_text = f'Session Accuracy: {avg_percentage:.1f}% ({self.splash_shots}/{self.total_shots})'
        self.avg_display.set_text(avg_text)
        
        # Create new scatter plot
        # self.scatter = self.ax.scatter(x_coords, y_coords, c=colors, cmap='coolwarm', edgecolors='none')
        self.scatter = plt.scatter(splash_xs, splash_ys, c='blue', s=200, marker='*', edgecolors='white', linewidths=0.75, zorder=4)
        # in points
        self.scatter = plt.scatter(in_xs, in_ys, c='green', s=100, marker='o', edgecolors='black', zorder=4)
        # out points
        self.scatter = plt.scatter(out_xs, out_ys, c='red', s=100, marker='X', edgecolors='black', zorder=4)
        
        # Redraw
        self.fig.canvas.draw()
        plt.pause(0.1)
        
    def start(self):
        """
        Start the interactive plotting
        """
        plt.ion()  # Turn on interactive mode
        plt.show()    
    
    def onclick(self, event):
        if event.inaxes:
            self.add_point(round(event.xdata, 2), round(event.ydata, 2))

plotter = RealTimePlotter()
plotter.start()