import openai
import os
import json
import requests
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI
from urllib.parse import quote

# Load environment variables from .env file
load_dotenv()


class OpenAIClientWithMemoryAndTools:
    
    # Loads your OpenAI key (either from .env or passed directly).
    # Creates a client to talk to the API.
    # Initializes empty conversation_history so the model can remember context across turns.
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=api_key)
        # Initialize conversation history in memory
        self.conversation_history: List[Dict[str, str]] = []
        
        # storing data (temporarly until i configure db)
        self.past_reviews = {
            "101A":[
                {
                    "movie":"Tarzan: The wonder car",
                    "review":"a bit boring"
                },
                {
                    "movie":"John Wick",
                    "review":"amazing,great stunts"
                },
                {
                    "movie":"Bellarina",
                    "review":"amazing,a great watch!"
                }
            ],
            "102B":[
                {
                    "movie":"Avengers : Infinity War",
                    "review":"a great movie"
                },
                {
                    "movie":"Jumanji: Welcome to jungle",
                    "review":"good movie"
                },
                {
                    "movie":"DDLJ",
                    "review":"boring"
                }
            ],
            "103":[
                {
                    "movie":"Star Wars",
                    "review":"a great movie"
                },
                {
                    "movie":"A quiet place",
                    "review":"good movie"
                },
            ]
        }
        
        self.genre_based_movies = {
            "Action":[
                "John Wick",
                "Rowdy Rathore",
                "Bellarina",
                "Avengers : Infinity War",
                "Avengers : End Game",
            ],
            "Adventure":[
                "Jumanji : Welcome to jungle",
                "Jumanji : Next level",
                "Avengers : End Game",
            ],
            "Romance":[
                "DDLJ",
                "Jab tak hey jaan",
                "Raja Hindustani",
                "Tarzan: The wonder car"
            ]
        }


    def fetch_past_reviews(self, user_id:str):
        print("fetching past reviews......",self.past_reviews.get(user_id, [])," ",user_id ,"\n")
        return self.past_reviews.get(user_id, [])
    
    def fetch_movies_genre(self, movie_name:str):
        genre_of_movie = []
        print("fetching genre's of liked movies......",genre_of_movie,"\n")
        for genre in self.genre_based_movies:
            if movie_name in self.genre_based_movies.get(genre,[]):
                # if genre not in genre_of_movie:
                #     genre_of_movie[genre] = []
                genre_of_movie.append(genre)
        
        return genre_of_movie
    
    def movies_with_genre(self, liked_genre:List = []):
        movies = {}
        print("finding the movies with liked genre........")
        for genre in self.genre_based_movies:
            if genre in liked_genre:
                for movie in self.genre_based_movies.get(genre,[]):
                    if movie not in movies:
                        movies[movie] = []
                    movies[movie].append(genre)
        
        print(movies,"\n")
        
        return movies
    
    
    # tools ends here
    def chat_completion_with_tools(self, user_message: str, system_message: Optional[str] = None):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "fetch_past_reviews",
                    "description": "Fetches past reviews of movies watched by user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "user_id of the user for which past reviews are needed to be fetched"
                            }
                        },
                        "required": ["user_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_movies_genre",
                    "description": "Fetches genres of the movie",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "movie_name": {
                                "type": "string",
                                "description": "name of movie for which genres are needed to be fetched"
                            }
                        },
                        "required": ["movie_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "movies_with_genre",
                    "description": "Fetches movie names that fall inside the requested genres",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "liked_genre": {
                                "type": "array",                # ✅ array
                                "items": {"type": "string"},    # ✅ each item is a string
                                "description": "List of genres for which movie names are needed"
                            }
                        },
                        "required": ["liked_genre"]
                    }
                }
            }

            
            
        ]
         # your schemas
        available_functions = {
            "fetch_past_reviews": self.fetch_past_reviews,
            "fetch_movies_genre": self.fetch_movies_genre,
            "movies_with_genre": self.movies_with_genre,
        }

        while True:  # loop until no more tool calls
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            response_message = response.choices[0].message
            print("="*80)
            print("🤖 LLM Response:", response_message)
            print("="*80)

            # If LLM makes a tool call
            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]  # take one at a time (sequential!)
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"🔧 Calling {fn_name} with {args}")

                func = available_functions.get(fn_name)
                if not func:
                    return f"⚠️ Unknown tool {fn_name}"

                result = func(**args)

                # Add assistant’s decision + tool’s response to messages
                messages.append({
                    "role": "assistant",
                    "content": response_message.content or "",
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

                # Continue loop → feed result back to LLM so it can decide next tool
                continue

            else:
                # No more tools requested → final answer
                ai_response = response_message.content.strip()
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": ai_response})
                return ai_response


    # def chat_completion_with_tools(self, 
    #                               user_message: str, 
    #                               system_message: Optional[str] = None,
    #                               max_tokens: int = 200,
    #                               temperature: float = 0.7) -> str:
    #     """
    #     Make a chat completion request with tool calling capabilities.
    #     """
    #     # Build messages list with full conversation history
    #     messages = []
        
    #     # Add system message if provided
    #     if system_message:
    #         messages.append({"role": "system", "content": system_message})
        
    #     # Add all previous conversation messages
    #     messages.extend(self.conversation_history)
        
    #     # Add current user message
    #     messages.append({"role": "user", "content": user_message})
        
    #     # Define available tools
    #     tools = [
    #         {
    #             "type": "function",
    #             "function": {
    #                 "name": "fetch_past_reviews",
    #                 "description": "Fetches past reviews of movies watched by user",
    #                 "parameters": {
    #                     "type": "object",
    #                     "properties": {
    #                         "user_id": {
    #                             "type": "string",
    #                             "description": "user_id of the user for which past reviews are needed to be fetched"
    #                         }
    #                     },
    #                     "required": ["user_id"]
    #                 }
    #             }
    #         },
    #         {
    #             "type": "function",
    #             "function": {
    #                 "name": "fetch_movies_genre",
    #                 "description": "Fetches genres of the movie",
    #                 "parameters": {
    #                     "type": "object",
    #                     "properties": {
    #                         "movie_name": {
    #                             "type": "string",
    #                             "description": "name of movie for which genres are needed to be fetched"
    #                         }
    #                     },
    #                     "required": ["movie_name"]
    #                 }
    #             }
    #         },
    #         # {
    #         #     "type": "function",
    #         #     "function": {
    #         #         "name": "movies_with_genre",
    #         #         "description": "fetches movie names which fall inside the requested genre's",
    #         #         "parameters": {
    #         #             "type": "object",
                        
    #         #             "properties": {
    #         #                 "liked_genre": {
    #         #                     "type": "json",
    #         #                     "description": "list of genre's for which movies names needed to be fetched"
    #         #                 }
    #         #             },
    #         #             "required": ["liked_genre"]
    #         #         }
    #         #     }
    #         {
    #             "type": "function",
    #             "function": {
    #                 "name": "movies_with_genre",
    #                 "description": "Fetches movie names that fall inside the requested genres",
    #                 "parameters": {
    #                     "type": "object",
    #                     "properties": {
    #                         "liked_genre": {
    #                             "type": "array",                # ✅ array
    #                             "items": {"type": "string"},    # ✅ each item is a string
    #                             "description": "List of genres for which movie names are needed"
    #                         }
    #                     },
    #                     "required": ["liked_genre"]
    #                 }
    #             }
    #         }

            
            
    #     ]
        
    #     available_functions = {
    #         "fetch_past_reviews": self.fetch_past_reviews,
    #         "fetch_movies_genre": self.fetch_movies_genre,
    #         "movies_with_genre": self.movies_with_genre,
    #     }
        
    #     try:
    #         while True:
    #             response = self.client.chat.completions.create(
    #                 model=self.model,
    #                 messages=messages,
    #                 tools=tools,
    #                 tool_choice="auto",
    #                 max_tokens=max_tokens,
    #                 temperature=temperature
    #             )
    #             response_message = response.choices[0].message
    #             print("="*100)
    #             print(response_message)
    #             print("="*100)
    #             # Check if the model wants to call a tool
    #             if response_message.tool_calls:
    #                 # Handle tool calls
    #                 # self.conversation_history.append({"role": "user", "content": user_message})

    #                 # Process all tool calls and collect results
    #                 tool_results = []
    #                 all_tool_calls = response_message.tool_calls
                    
    #                 # Add assistant message with all tool calls to messages (for final API call)
    #                 messages.append({
    #                     "role": "assistant", 
    #                     "content": response_message.content or "",
    #                     "tool_calls": all_tool_calls
    #                 })
                    
    #                 print(all_tool_calls,"_____________________________________")
    #                 for tool_call in all_tool_calls :
    #                     # tool_call = all_tool_calls[0]
    #                     fn_name = tool_call.function.name
    #                     args = json.loads(tool_call.function.arguments)
    #                     print(f"🔧 Calling {fn_name} with {args}")
                        
    #                     # Run the tool
    #                     func = available_functions.get(fn_name)
    #                     if func:
    #                         result = func(**args)
    #                         tool_results.append({
    #                             "tool_call_id": tool_call.id,
    #                             "content": json.dumps(result)
    #                         })
    #                     else:
    #                         print(f"⚠️ Unknown function {fn_name}")
    #                     tool_results.append(result)
                    
    #                     # Add all tool results to messages for the final API call (after the loop)
    #                     for result in tool_results:
    #                         messages.append({
    #                             "role": "tool",
    #                             "tool_call_id": result["tool_call_id"],
    #                             "content": result["content"]
    #                         })
                        
    #                     # Add the grouped tool calls to conversation history (single entry)
    #                     self.conversation_history.append({
    #                         "role": "assistant", 
    #                         "content": response_message.content or "",
    #                         "tool_calls": all_tool_calls
    #                     })
                    
    #                     # Add the grouped tool results to conversation history (single entry)
    #                     self.conversation_history.append({
    #                         "role": "tool",
    #                         "tool_calls": all_tool_calls,
    #                         "tool_results": tool_results
    #                     })
                        
    #                 continue    
                    
    #             else:
    #                 # No tool calls, normal response
    #                 ai_response = response_message.content.strip()
                    
    #                 # Update conversation history
    #                 self.conversation_history.append({"role": "user", "content": user_message})
    #                 self.conversation_history.append({"role": "assistant", "content": ai_response})
                    
    #                 return ai_response
            
    #     except Exception as e:
    #         return f"Error making API call: {str(e)}"

    def start_conversation(self):
        """
        Start a continuous conversation loop with memory and tool calling.
        """
        print("🤖 Chatbot with Memory and Tools is ready!")
        print("=" * 60)
        # print("Available tools: get_weather (try asking about weather!)")
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'history' to see conversation history")
        print("Type 'clear' to clear conversation history")
        print("=" * 60)
        
        system_message = """You are a helpful AI assistant with access to tools. 
        Ask user it's user_id if user doesn't mention.
        You are movie recommender bot which recommends movies based on the past reviews of movie.
        You can get past reviews of movies which were watched by user using the fetch_past_reviews tool.
        You make a list of movie names which were liked by user based on the past reviews of movies of user.
        Than You extract the genre of movie which were liked by user using fetch_movie_genre.
        later you fetch the movies of same genre which were liked by user using movies_with_genre. 
        You recommend one movie which is more comman with the liked movies
        Keep responses concise and engaging. 
        """
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 AI: Goodbye! Thanks for chatting!")
                    break
                
                # Check for special commands
                if user_input.lower() == 'history':
                    self.show_conversation_history()
                    continue
                
                if user_input.lower() == 'clear':
                    self.clear_conversation_history()
                    continue
                
                if not user_input:
                    print("Please enter a message.")
                    continue
                
                # Send to LLM with tool calling capabilities
                print("🔄 Processing...")
                response = self.chat_completion_with_tools(
                    user_message=user_input,
                    system_message=system_message
                )
                
                # Display response
                print(f"🤖 AI: {response}")
                
            except KeyboardInterrupt:
                print("\n\n🤖 AI: Conversation interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    

# Example usage and demo
if __name__ == "__main__":
    try:
        client = OpenAIClientWithMemoryAndTools()
        client.start_conversation()
        
        # print(client.fetch_past_reviews(user_id),"\n")
        # print(client.fetch_movies_genre(movie_name="Avengers : End Game"),"\n")
        # print(client.movies_with_genre(liked_genre = ["Action","Romance"]))
        # client = OpenAIClientWithMemoryAndTools()
        # client.start_conversation()
    except ValueError as e:
        print(f"Setup error: {e}")
        print("Please set your OPENAI_API_KEY environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nNote: Weather data is provided by wttr.in (free, no API key required)")
    except Exception as e:
        print(f"Unexpected error: {e}")
    


# *Homework* – Multi-hop Tool Calling (Movie Recommender Agent)

# Build a very simple simulation of multi-hop tool calling.
# The idea is that the LLM should not just call one tool and stop, but instead demonstrate multi-hop tool calling (not just single hop).

# We’ll simulate this with a movie recommender agent using in-memory data structures (no persistence, no complex engineering).

# Flow looks like this:
# 1. *User* : “I want to watch some movies.”
# *LLM* : “Please provide your user ID.”
# 2. *User gives ID* (e.g., 101).
# *LLM → Tool* : fetch past reviews of this user.
# *Tool → LLM* : returns list of past movies + reviews.
# 3. *LLM → Tool* : fetch genres of movies the user liked.
# *Tool → LLM* : returns union of genres.
# 4. *LLM → Tool* : fetch new movies in those genres, excluding already watched.
# *Tool → LLM* : returns candidate movies.
# 5. *LLM Final Response* : “I recommend you watch Movie A and Movie B.”
# tools