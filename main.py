from dotenv import load_dotenv
from graph.graph import graph

load_dotenv()
if __name__=="__main__":
    print("Hello Advanced RAG!")
    # res1    = graph.invoke(input={"question": "What is agent memory ?"})
    # print(f"Answer 1 = {res1}")
    print("-"*20)
    res2    = graph.invoke(input={"question": "How fast if Mig 31. What is its service ceiling ?"})
    print(f"\n\nAnswer 2 = {res2}")
