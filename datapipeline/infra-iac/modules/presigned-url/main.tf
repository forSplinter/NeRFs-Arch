resource "aws_api_gateway_rest_api" "nerfarc_api_presigned_url" {
  name = var.nerfarc_api_gtw_name
  endpoint_configuration {
    types = ["REGIONAL"]  
  }
  tags = {
    Name = "${var.env}_${var.nerfarc_api_gtw_name}"
    env  = var.env
  }
}

resource "aws_api_gateway_resource" "generate_url" {
  rest_api_id = aws_api_gateway_rest_api.nerfarc_api_presigned_url.id
  parent_id   = aws_api_gateway_rest_api.nerfarc_api_presigned_url.root_resource_id
  path_part   = "generate-url"  
}

resource "aws_api_gateway_method" "generate_url_method" {
  rest_api_id   = aws_api_gateway_rest_api.nerfarc_api_presigned_url.id
  resource_id   = aws_api_gateway_resource.generate_url.id  
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda_integration" {
  rest_api_id = aws_api_gateway_rest_api.nerfarc_api_presigned_url.id
  resource_id = aws_api_gateway_resource.generate_url.id
  http_method = aws_api_gateway_method.generate_url_method.http_method
  
  integration_http_method = "POST"
  type                    = "AWS_PROXY"  //see y later
  uri                     = aws_lambda_function.presigned_lambda.invoke_arn //need to create the lambda first  
}

resource "aws_api_gateway_method_response" "proxy" {
  rest_api_id = aws_api_gateway_rest_api.nerfarc_api_presigned_url.id
  resource_id = aws_api_gateway_resource.generate_url.id
  http_method = aws_api_gateway_method.generate_url_method.http_method
  status_code = "200"
  
  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = true
  }
}

resource "aws_api_gateway_integration_response" "proxy" {
  rest_api_id = aws_api_gateway_rest_api.nerfarc_api_presigned_url.id
  resource_id = aws_api_gateway_resource.generate_url.id
  http_method = aws_api_gateway_method.generate_url_method.http_method
  status_code = aws_api_gateway_method_response.proxy.status_code

  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = "'*'"
  }

  depends_on = [aws_api_gateway_integration.lambda_integration]
}

resource "aws_api_gateway_method" "options_method" {
  rest_api_id   = aws_api_gateway_rest_api.nerfarc_api_presigned_url.id
  resource_id   = aws_api_gateway_resource.generate_url.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "options_integration" {
  rest_api_id = aws_api_gateway_rest_api.nerfarc_api_presigned_url.id
  resource_id = aws_api_gateway_resource.generate_url.id
  http_method = aws_api_gateway_method.options_method.http_method
  
  type = "MOCK"
  
  request_templates = {
    "application/json" = jsonencode({ statusCode = 200 })
  }
}

resource "aws_api_gateway_method_response" "options_200" {
  rest_api_id = aws_api_gateway_rest_api.nerfarc_api_presigned_url.id
  resource_id = aws_api_gateway_resource.generate_url.id
  http_method = aws_api_gateway_method.options_method.http_method
  status_code = "200"
  
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true,
    "method.response.header.Access-Control-Allow-Methods" = true,
    "method.response.header.Access-Control-Allow-Origin"  = true
  }
}

resource "aws_api_gateway_integration_response" "options_integration_response" {
  rest_api_id = aws_api_gateway_rest_api.nerfarc_api_presigned_url.id
  resource_id = aws_api_gateway_resource.generate_url.id
  http_method = aws_api_gateway_method.options_method.http_method
  status_code = "200"
  
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'",
    "method.response.header.Access-Control-Allow-Methods" = "'POST,OPTIONS'",
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }
}

resource "aws_api_gateway_deployment" "deployment" {
  depends_on = [
    aws_api_gateway_integration.lambda_integration,
    aws_api_gateway_integration.options_integration
  ]
  
  rest_api_id = aws_api_gateway_rest_api.nerfarc_api_presigned_url.id
}