#include <stdio.h>
#include <iostream>
#include <edelweiss/JsonParser.h>
#include <fstream>
#include <string.h>
//#include <edelweiss/LibCurl.h>

using namespace std;

double GetDouble(JsonObject jObject) {
	return atof(jObject.toString().c_str());
}

int find_value (JsonObject jObject, string key , string value , double num)
{
  vector<JsonObject> arr = jObject[key][value];
  for(size_t itr = 0 ; itr < arr.size() ; itr++)
  {
    if(GetDouble(arr[itr]) == num)
      return itr;
  } 
  return -1;
}

int main ()
{
  std::ifstream inFile("2.txt", std::ifstream::binary);
	std::string fileContents((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());


	JsonParser parser(fileContents);

	JsonObject reqObject = parser.getJsonObject();

	JsonObject plotPoints = reqObject["data"]["pltPnts"];
	vector<JsonObject> script_open(plotPoints["open"]);
	vector<JsonObject> script_low(plotPoints["low"]);
	vector<JsonObject> script_high(plotPoints["high"]);
	vector<JsonObject> script_close(plotPoints["close"]);
  vector<JsonObject> script_vol(plotPoints["vol"]);
  vector<JsonObject> script_ltt(plotPoints["ltt"]);
  
  cout<<script_low.size()<<endl;
  cout<<GetDouble(script_open[4])<<endl;
  ofstream outfile;
  outfile.open("str.txt");
    
  outfile<<"open\tclose\tlow\thigh\tvol\tltt"<<endl;
    
  for(size_t iter = 0 ; iter < script_low.size() ; iter++)
  {
      outfile<<GetDouble(script_open[iter])<<"\t";
      outfile<<GetDouble(script_close[iter])<<"\t";
      outfile<<GetDouble(script_low[iter])<<"\t";
      outfile<<GetDouble(script_high[iter])<<"\t";
      outfile<<GetDouble(script_vol[iter])<<"\t";
      outfile<<script_ltt[iter].toString().c_str()<<"\t"<<endl;
  }
  
  
  
  outfile.close();
  
  return 0;
    
}
