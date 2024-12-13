import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mplPath
import matplotlib.animation as animation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from collections import deque

def quantized_2d_gaussian(width, sigma=0.5):
    # Create a grid of (x, y) coordinates
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, width)
    x, y = np.meshgrid(x, y)
    
    # Define the 2D Gaussian function
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize the values to range from 0 to 1
    gaussian_normalized = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())
    
    return gaussian_normalized

def add_gaussian_to_array(array_size, x, y, gaussian_width=5):
    # Create a 2D numpy array of zeroes
    array = np.zeros((array_size, array_size))
    
    # Generate the 2D Gaussian
    gaussian = quantized_2d_gaussian(gaussian_width)
    half_size = gaussian.shape[0] // 2
    
    # Convert the (x, y) coordinates to indices in the array
    x_idx = np.searchsorted(np.linspace(-20, 20, array_size), x)
    y_idx = np.searchsorted(np.linspace(-20, 20, array_size), y)
    
    # Add the Gaussian to the array at the specified location
    for i in range(-half_size, half_size):
        for j in range(-half_size, half_size):
            if 0 <= x_idx + i < array_size and 0 <= y_idx + j < array_size:
                array[y_idx + j, x_idx + i] += gaussian[j + half_size, i + half_size]
    
    return array

def visualize_gaussian_addition(array_size=20, x=0, y=0, gaussian_width=5):
    # Create the Gaussian array
    gaussian_array = add_gaussian_to_array(array_size, x, y, gaussian_width)
    
    # Create a figure
    fig, ax = plt.subplots()
    
    # Display the array as an image
    extent = [-20, 20, -20, 20]
    ax.set_xticks(range(-20, 21, 2))
    ax.set_yticks(range(-20, 21, 2))
    ax.grid(True)
    im = ax.imshow(gaussian_array, extent=extent, origin='lower', cmap='plasma')
    
    # Add colorbar
    plt.colorbar(im)
    
    # Add title and labels
    plt.title('2D Gaussian Addition Visualization')
    plt.xlabel('X-Axis (cm)')
    plt.ylabel('Z-Axis (cm)')
    
    plt.show()

# visualize_gaussian_addition(x=11, y=-13, gaussian_width=10)


class RealTimePlotter:
    def reset(self):
        self.x, self.y = None, None

    def __init__(self):
        self.x, self.y = None, None

        # Create a figure and two subplots side by side
        self.fig, (self.shotmap, self.heatmap) = plt.subplots(1, 2, figsize=(20, 10))

        # Set the limits for both axes
        self.shotmap.set_xlim([-20, 20])
        self.shotmap.set_ylim([-20, 20])
        self.heatmap.set_xlim([-20, 20])
        self.heatmap.set_ylim([-20, 20])

        # Set grid lines every 2 units
        self.shotmap.set_xticks(range(-20, 21, 2))
        self.shotmap.set_yticks(range(-20, 21, 2))
        self.heatmap.set_xticks(range(-20, 21, 2))
        self.heatmap.set_yticks(range(-20, 21, 2))

        # Enable grid
        self.shotmap.grid(True)
        self.heatmap.grid(True)

        # automatically remove old gaussians when new ones are added
        # saves the past 10 shots
        self.recent_len = 10
        self.gauss_queue = deque(maxlen=self.recent_len)

        # Create a 2D numpy array for the heatmap data
        self.hmres = 80 # heatmap resolution
        self.gauss_width = 20
        # self.heatmap_data = np.zeros((self.hmres, self.hmres))  # Example data, replace with your actual data
        self.heatmap_data = np.random.rand(self.hmres, self.hmres)  # Example data, replace with your actual data

        # Display the heatmap over the self.heatmap graph
        self.extent = [-20, 20, -20, 20]  # Define the extent of the heatmap
        self.im = self.heatmap.imshow(self.heatmap_data, extent=self.extent, origin='lower', cmap='plasma', alpha=0.5)

        # Add major gridlines along x=0 and y=0 for both axes
        self.shotmap.axhline(0, color='black', linewidth=1.5)
        self.shotmap.axvline(0, color='black', linewidth=1.5)
        self.heatmap.axhline(0, color='black', linewidth=1.5)
        self.heatmap.axvline(0, color='black', linewidth=1.5)

        # Add a title to the entire figure
        self.fig.suptitle('Splash Metrics & Visual Feedback', fontsize=16)

        # Add titles to the subplots
        self.shotmap.set_title('Shot Map')
        self.heatmap.set_title('Heat Map')

        # Add labels to the x and 'z' axes
        self.shotmap.set_xlabel('X-Axis (cm)')
        self.shotmap.set_ylabel('Z-Axis (cm)')
        self.heatmap.set_xlabel('X-Axis (cm)')
        self.heatmap.set_ylabel('Z-Axis (cm)')

        # Add the square and circle patches to both shotmap and heatmap
        # shotmap
        self.sm_square = self.square()
        self.sm_circle = self.circle()
        # heatmap
        self.hm_square = patches.Rectangle(
            (-10, -10),  # Lower-left corner
            20,          # Width (from -10 to 10)
            20,          # Height (from -10 to 10)
            alpha=0.3,   # Transparency (0.0 to 1.0)
            facecolor='none',  # No fill color (transparent)
            edgecolor='black',  # Border color
            linewidth=2,  # Line thickness
            zorder=3     # Ensure square is drawn above grid lines
        )
        self.hm_circle = patches.Circle(
            (0, 0), # Centered at 0, 0
            radius=4.6,  # cup radius of 4.6cm --> diameter = 9.2cm == 92mm
            alpha=0.3,   # Transparency (0.0 to 1.0)
            facecolor='none',  # No fill color (transparent)
            edgecolor='black',  # Border color
            linewidth=2,  # Line thickness
            zorder=4     # Ensure square is drawn above grid lines
        )

        self.shotmap.add_patch(self.sm_square)
        self.shotmap.add_patch(self.sm_circle)
        self.heatmap.add_patch(self.hm_square)
        self.heatmap.add_patch(self.hm_circle)

        # Create a Path object for the square & circle to check point containment
        # only can add points on the shotmap
        self.square_vertices = [(-10, -10), (-10, 10), (10, 10), (10, -10)]
        self.square_path = mplPath.Path(self.square_vertices)
        self.circle_path = mplPath.Path.circle(radius=4.6)

        # Initialize empty scatter plots for both maps
        self.shot_scatter = self.shotmap.scatter([], [], color='red', s=50, zorder=5)

        # Connect the mouse click event to the plotting function
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # Initialize point storage
        self.points = []

        # Session metrics
        self.total_shots = 0
        self.splash_shots = 0
        self.session_percent = 0.0
        self.session_percent_text = f' Session Accuracy: {self.session_percent:.1f}% ({self.splash_shots}/{self.total_shots})'

        self.recent_shots = deque(maxlen=self.recent_len)
        self.recent_percent = 0.0
        self.recent_percent_text = f'Recent Accuracy: {self.session_percent:.1f}% ({np.sum(self.recent_shots)}/{len(self.recent_shots)})'
        
        # Add text annotation for accuracy between the two subplots
        self.avg_display = self.fig.text(0.5, 0.05, self.session_percent_text, 
            horizontalalignment='center', verticalalignment='center', fontsize=12)
        self.recent_display = self.fig.text(0.5, 0.02, self.recent_percent_text,
            horizontalalignment='center', verticalalignment='center', fontsize=12)

        # Set aspect ratio to be equal. Thus when resizing window, graphs stay square.
        self.shotmap.set_aspect('equal', adjustable='box')
        self.heatmap.set_aspect('equal', adjustable='box')

        legend_elements = [self.sm_square, self.sm_circle]
        # Create legend handles with labels
        legend_labels = ['Catch Zone', 'Splash Zone']
        self.fig.legend(legend_elements, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
                        ncol=2, fancybox=True, shadow=True, framealpha=1)


    def add_point(self, x, y):
        point = (x, y)
        self.points.append(point)

        # Convert the point coordinates to indices in the heatmap grid
        gaussian = add_gaussian_to_array(self.hmres, x, y, gaussian_width=self.gauss_width)
        self.gauss_queue.append(gaussian)

        # sum up last ten gaussians elementwise
        gauss_sum = np.sum(self.gauss_queue, axis=0)
        self.heatmap_data = gauss_sum

        # Normalize the heatmap data such that the highest value is 1
        if self.heatmap_data.max() > 0:
            self.heatmap_data = self.heatmap_data / self.heatmap_data.max()

        # Update session metrics
        self.total_shots += 1
        if self.circle_path.contains_point(point): 
            self.splash_shots += 1
            self.recent_shots.append(1)
        else:
            self.recent_shots.append(0)        

        # Update the plot after 
        self.update_plot()

    # Updates based on internal saved x and y values given by the cameras
    def update_saved(self):
        if self.x != None and self.y != None:
            print(f"Visualizing {self.x}, {self.y}")
            self.add_point(self.x, self.y)
            self.reset() # reset x and y to None

    def onclick(self, event):
        if event.inaxes:
            x, y = round(event.xdata, 2), round(event.ydata, 2)
            self.add_point(x, y)

    # Update the scatter plot with current points
    def update_plot(self):
        # Clear previous scatter plot
        self.shot_scatter.remove()
        # self.heat_scatter.remove()
        if not self.points:
            return
        
        # Update the heatmap visualization
        # self.heatmap_data = np.random.rand(self.hmres, self.hmres)  # Example data, replace with your actual data
        self.im.set_data(self.heatmap_data)
        self.fig.canvas.draw_idle()

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

        session_percent = (self.splash_shots / self.total_shots) * 100 if self.total_shots > 0 else 0
        session_percent_text = f'Session Accuracy: {session_percent:.1f}% ({self.splash_shots}/{self.total_shots})'
        self.avg_display.set_text(session_percent_text)

        self.recent_percent = (np.sum(self.recent_shots) / len(self.recent_shots)) * 100
        self.recent_percent_text = f'Recent Accuracy: {self.recent_percent:.1f}% ({np.sum(self.recent_shots)}/{len(self.recent_shots)})'
        self.recent_display.set_text(self.recent_percent_text)
        
        
        # Create new scatter plot
        self.shot_scatter = self.shotmap.scatter(splash_xs, splash_ys, c='blue', s=200, marker='*', edgecolors='white', linewidths=0.75, zorder=4)
        # in points
        self.shot_scatter = self.shotmap.scatter(in_xs, in_ys, c='green', s=100, marker='o', edgecolors='black', zorder=4)
        # out points
        self.shot_scatter = self.shotmap.scatter(out_xs, out_ys, c='red', s=100, marker='X', edgecolors='black', zorder=4)
        
        # Redraw
        self.fig.canvas.draw()
        plt.pause(0.1)
        

    # Create the square representing the robot's area of correction
    def square(self):
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
        return square

    def circle(self):
        circle = patches.Circle(
            (0, 0), # Centered at 0, 0
            radius=4.6,  # cup radius of 4.6cm --> diameter = 9.2cm == 92mm
            facecolor='blue',  # Fill color
            alpha=0.3,   # Transparency (0.0 to 1.0)
            edgecolor='blue',  # Border color
            linewidth=2,  # Line thickness
            zorder=4     # Ensure square is drawn above grid lines
        )
        return circle

    def show(self):
        # Show the plot
        plt.ion()
        # self.fig.canvas.draw_idle()
        self.fig.canvas.draw()
        plt.pause(0.1)
        # plt.show()


# plotter = RealTimePlotter()
# plotter.show()

