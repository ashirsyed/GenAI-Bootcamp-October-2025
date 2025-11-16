from langgraph.graph import StateGraph, START, END
from blogstate import BlogState
from blog_node import BlogNode

class GraphBuilder:
    def __init__(self,llm):
        self.llm=llm
        self.graph=StateGraph(BlogState)

    def build_topic_graph(self):
        """
        Build a graph to generate blogss based on topic
        """
        self.blog_node_obj=BlogNode(self.llm)
        
        ## Nodes
        self.graph.add_node("title_creation", self.blog_node_obj.title_creation)
        self.graph.add_node("content_generation",self.blog_node_obj.content_generation)
        self.graph.add_node("audio_generation",self.blog_node_obj.audio_generation)
        self.graph.add_node("photo_generation",self.blog_node_obj.photo_generation)

        ## Edges
        self.graph.add_edge(START,"title_creation")
        self.graph.add_edge("title_creation","content_generation")
        self.graph.add_edge("content_generation","audio_generation")
        self.graph.add_edge("audio_generation","photo_generation")
        self.graph.add_edge("photo_generation",END)

        return self.graph
    
    def setup_graph(self,usecase):
        if usecase=="topic":
            self.build_topic_graph()

        return self.graph.compile()
    