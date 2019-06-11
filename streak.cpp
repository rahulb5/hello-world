#include <stdio.h>
#include <iostream>
#include <edelweiss/JsonParser.h>
#include <fstream>
#include <string.h>

using namespace std;


//declaring class to store variables of trade
class trade
{
  public: 
          vector<int> transaction;
          vector<int> investment;
          vector<int> sell;
          vector<string> entry_date;
          vector<string> exit_date;
          int pos, neg , profit;
          
          void summary()
          {
            
            cout<<"Total Profit: "<<profit<<endl;
            cout<<"Successful trades: "<<pos<<endl;
            cout<<"Unsuccessful trades: "<<neg<<endl; 
            cout<<"Success ratio: "<<(float)pos/(pos+neg)<<endl ;
          }
          
          //constructor
          trade()
          {
            pos = 0;
            neg = 0;
            profit = 0;
          }
};

class query
{
  public: 
          int first_parameter , condition , second_parameter;
          
          query()
          {
            first_parameter = 0;
            condition = 0;
            second_parameter = 0;
          }
};
double GetDouble(JsonObject jObject) {
	return atof(jObject.toString().c_str());
}


string assign (int a)
{
  if(a == 1)
    return "open";
  else if(a==2)
    return "close";
  else if (a==3)
    return "high";
  else if (a==4)
    return "low";
  else if (a==5)
    return "1";
  else 
    return "2";
}

/*
  1.greater than
  2.less than
  3.cross above
  4.cross below
  5.equal to
*/


int condition (int cond, double price , double prev_price, double ind_point , double prev_ind_point) 
{
  if(cond == 1)
  {
    if(price > ind_point)
      return 1;
    else
      return 0;
  }  
  else if(cond == 2)
  {
    if(price < ind_point)
      return 1;
    else
      return 0;
  }
  else if(cond == 3)
  {
    if(price > ind_point && prev_price < prev_ind_point)
      return 1;
    else
      return 0;
  }
  else if(cond == 4)
  {
    if(price < ind_point && prev_price > prev_ind_point)
      return 1;
    else
      return 0;
  }
  else if(cond == 5)
  {
    if(price == ind_point)
      return 1;
    else
      return 0;
  } 
}



trade backtest(vector<query> user_entry_cond , vector<query> user_exit_cond , vector<vector<JsonObject> > Json_datapoints , int qty = 1)
{
  int first_para , prev_first_para , second_para ,prev_second_para;
  int flag_entry_cond = 0 , j = 0;
  int i = 1;
  //ofstream dates;
  //dates.open("entry_dates.txt");
 while(i < Json_datapoints[0].size())
 { 
  // cout<<i<<" ";
   flag_entry_cond = 0;  
   for(j = 0 ; j < user_entry_cond.size() ; j++)
   {
     first_para = GetDouble(Json_datapoints[user_entry_cond[j].first_parameter-1][i]);
     prev_first_para = GetDouble(Json_datapoints[user_entry_cond[j].first_parameter-1][i-1]);
     second_para = GetDouble(Json_datapoints[user_entry_cond[j].second_parameter-1][i]);
     prev_second_para = GetDouble(Json_datapoints[user_entry_cond[j].second_parameter-1][i-1]);
     
     if(!condition(user_entry_cond[j].condition , first_para , prev_first_para , second_para , prev_second_para))
     {
       flag_entry_cond = 1;
       break;
     } 
   }
   
   if(flag_entry_cond == 0)
   {
     cout<<GetDouble(Json_datapoints[1][i])<<"\t"<<Json_datapoints.back()[i].toString().c_str()<<"\t"<<i<<"\t"<<j<<endl;
   } 
   //cout<<i<<endl;
   i++;
 } 
 cout<<"problem";
 //dates.close();
}

int main ()
{
  std::ifstream inFile("3.txt", std::ifstream::binary);
	std::string fileContents((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());

  //making JSON object from the data given
	JsonParser parser(fileContents);
	JsonObject reqObject = parser.getJsonObject();
  JsonObject techind = reqObject["data"]["tiOut"];
  
  //extracting JSON object from the above JSON object which contains only the trade data
	JsonObject plotPoints = reqObject["data"]["pltPnts"];
 
  //making vector to get individual data point (still a JSON object)
  
  vector<vector<JsonObject> > Json_datapoints;
  
	Json_datapoints.push_back(plotPoints["open"]);
	Json_datapoints.push_back(plotPoints["close"]);
  Json_datapoints.push_back(plotPoints["high"]);
  Json_datapoints.push_back(plotPoints["low"]);
  
  
  
	Json_datapoints.push_back(techind[0]["rsltSet"][0]["vals"]);
	Json_datapoints.push_back(techind[1]["rsltSet"][0]["vals"]);
  Json_datapoints.push_back(plotPoints["vol"]);
  Json_datapoints.push_back(plotPoints["ltt"]);
    
  
  ofstream temp;
  temp.open("temp.txt");
  int j;
  temp<<"open\tclose\t\thigh\t\tlow\t\t\tind1\t\tind2\t\tvol\t\tltt\n"; 
  for(int i = 0 ; i < Json_datapoints[0].size() ; i++)
  {
    for(j = 0 ; j < Json_datapoints.size() - 1 ; j++)
    {
      temp<<GetDouble(Json_datapoints[j][i])<<"  ";
    }
    temp<<Json_datapoints[j][i];
    temp<<endl;
  }
  
  temp.close();
  
  vector<query> user_entry_cond, user_exit_cond;
  query temp1;
  
  int flag = 0;
  
  cout<<"Enter entry condition: \n\n\n";
  
  do
  {
    user_entry_cond.push_back(temp1);
    cout<<"Enter first parameter: \n";
    cout<<"1.open\t2.close\t3.high\t4.low\t5.SMA 14\t6.SMA 100\n";
    cin>>user_entry_cond.back().first_parameter;
    cout<<"Enter condition: ";
    cout<<"1.greater than\n2.less than\n3.cross above\n4.cross below\n5.equal to\n";
    cin>>user_entry_cond.back().condition;
    cout<<"Enter second parameter: \n";
    cout<<"1.open\t2.close\t3.high\t4.low\t5.SMA 14\t6.SMA 100\n";
    cin>>user_entry_cond.back().second_parameter;
    cout<<"do you want to give more entry condition:\n1.Yes\n2.no\n";
    cin>>flag;  
  }while(flag == 1);
  
  cout<<"Enter exit condition: \n\n\n";
  
  do
  {
    user_exit_cond.push_back(temp1);
    cout<<"Enter first parameter: \n";
    cout<<"1.open\n2.close\n3.high\n4.low\n5.SMA 14\n6.SMA 100\n";
    cin>>user_exit_cond.back().first_parameter;
    cout<<"Enter condition: \n";
    cout<<"1.greater than\n2.less than\n3.cross above\n4.cross below\n5.equal to\n";
    cin>>user_exit_cond.back().condition;
    cout<<"Enter second parameter: \n";
    cout<<"1.open\n2.close\n3.high\n4.low\n5.SMA 14\n6.SMA 100\n";
    cin>>user_exit_cond.back().second_parameter;
    cout<<"do you want to give more exit condition:\n1.Yes\n2.no\n";
    cin>>flag;
    
  }while(flag == 1);
  
  
  
  
  
  
  
  
   
  int first_para , prev_first_para , second_para ,prev_second_para;
  int flag_entry_cond = 0 , qty =1 , flag_exit_cond = 0 , buy_flag = 0 , profit = 0;
  ofstream dates;
  dates.open("dates.txt");
  trade back;
  
  for(int i = 1 ; i < Json_datapoints[0].size() ; i++)
  { 
    if(buy_flag == 0)
    { 
      flag_entry_cond = 0;  
     for(j = 0 ; j < user_entry_cond.size() ; j++)
     {
       first_para = GetDouble(Json_datapoints[user_entry_cond[j].first_parameter-1][i]);
       prev_first_para = GetDouble(Json_datapoints[user_entry_cond[j].first_parameter-1][i-1]);
       second_para = GetDouble(Json_datapoints[user_entry_cond[j].second_parameter-1][i]);
       prev_second_para = GetDouble(Json_datapoints[user_entry_cond[j].second_parameter-1][i-1]);
     
       if(!condition(user_entry_cond[j].condition , first_para , prev_first_para , second_para , prev_second_para))
       {
         flag_entry_cond = 1;
         break;
       } 
     }
   
     if(flag_entry_cond == 0)
     {
       buy_flag = 1;
       back.investment.push_back(GetDouble(Json_datapoints[1][i])*qty);
       back.entry_date.push_back(Json_datapoints.back()[i].toString().c_str());
       dates<<GetDouble(Json_datapoints[1][i])<<"\t"<<Json_datapoints.back()[i].toString().c_str()<<"\t";
     } 
    }
    
    else
    {
      flag_exit_cond = 0;
      for(j = 0 ; j < user_exit_cond.size() ; j++)
      {
       first_para = GetDouble(Json_datapoints[user_exit_cond[j].first_parameter-1][i]);
       prev_first_para = GetDouble(Json_datapoints[user_exit_cond[j].first_parameter-1][i-1]);
       second_para = GetDouble(Json_datapoints[user_exit_cond[j].second_parameter-1][i]);
       prev_second_para = GetDouble(Json_datapoints[user_exit_cond[j].second_parameter-1][i-1]);
       if(!condition(user_exit_cond[j].condition , first_para , prev_first_para , second_para , prev_second_para))
       {
         flag_exit_cond = 1;
         break;
       }
      }
      if(flag_exit_cond == 0)
      {
        buy_flag = 0; 
        back.sell.push_back(qty*GetDouble(Json_datapoints[1][i]));
        profit = back.sell.back() - back.investment.back();
        back.profit += profit;
        back.transaction.push_back(profit);
        back.exit_date.push_back(Json_datapoints.back()[i].toString().c_str());
        if(profit>=0)
          back.pos++;
        else
          back.neg++;
        dates<<GetDouble(Json_datapoints[1][i])<<"\t"<<Json_datapoints.back()[i].toString().c_str()<<"\t"<<profit<<endl;
      }
      
    } 
  }
 if(buy_flag == 1)
 {
    cout<<"\nStanding trade: "<<back.investment.back()<<endl;
 } 
  
 dates.close();
 back.summary(); 
  return 0;
    
}
