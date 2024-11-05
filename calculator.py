def funcs():
	functions = [{
			"type": "function",
			"function": 
			{
				"name": "calculate",
				"description": "Performs a given mathematical operation including addition, subtraction, multiplication, division and brackets",
				"parameters": {
						"type": "object",
						"properties": {
							"calculate": {
								"type": "string",
								"description": "Mathematical expression to solve",
							},
						},
						"required": [],
					},
			}
		}
	]
	return functions