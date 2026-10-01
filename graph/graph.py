from langgraph.graph import StateGraph,END
from graph.state import FoodState
from graph.nodes import disambiguation_agent,nutrition_lookup,explain_nutrition,embed_image_node,retrieve_foods

graph=StateGraph(FoodState)


graph.add_node("embed",embed_image_node)
graph.add_node("retrieve",retrieve_foods)
graph.add_node("agent",disambiguation_agent)
#graph.add_node("ask_user",ask_user)
graph.add_node("nutrition",nutrition_lookup)
graph.add_node("explain",explain_nutrition)

graph.set_entry_point("embed")

graph.add_edge("embed","retrieve")

graph.add_edge('retrieve','agent')

graph.add_conditional_edges(
    "agent",
    lambda s:"nutrition" if "selected_food" in s else END,
    {
        "nutrition":"nutrition",
        END:END
    }
)

graph.add_edge("nutrition","explain")
graph.add_edge("explain",END)

food_graph= graph.compile()