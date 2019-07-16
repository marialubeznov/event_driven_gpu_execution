

#define NO_OF_NODES 65536

#define NO_OF_EDGES 393930

#define CHUNK_SIZE 2560
#define DIM 256


/* this structure represents a point */
/* these will be passed around to avoid copying coordinates */
typedef struct {
  float weight;
  float coord[DIM];
  long assign;  /* number of point where this one is assigned */
  float cost;  /* cost of that assignment, weight*distance */
} Point;

/* this is the array of points */
typedef struct {
  long num; /* number of points; may not be N if this is a sample */
  int dim;  /* dimensionality */
  Point p[CHUNK_SIZE]; /* the array itself */
  long kcenter;
  float totalcost;
  float z;
} Points;

void pspeedy(Points *points);
float dist(Point p1, Point p2, int dim);


struct Node
{
	int starting;
	int no_of_edges;
};


struct host_graph 
{
	Node h_graph_nodes[NO_OF_NODES];
	bool h_graph_mask[NO_OF_NODES];
	bool h_updating_graph_mask[NO_OF_NODES];
	bool h_graph_visited[NO_OF_NODES];
	int h_graph_edges[NO_OF_EDGES];
};



void init_graph_from_file(char* fname,host_graph* h_graph);

void read_sep( float* dest, int dim, int num, int n ) ;


