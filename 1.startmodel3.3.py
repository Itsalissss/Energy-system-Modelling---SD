import pandas as pd # useful for importing and exporting data to excel
import numpy as np  # useful for creating and editing (multidimensional) arrays

#%% Model Parameters
initialTime         = 0
finalTime           = 100
lengthTimeStep      = 1
timeSteps           = np.arange(initialTime,finalTime+lengthTimeStep,lengthTimeStep)

#%% Vensim Functions 



def PulseTrain(current,start,steps,end):
    """Returns a 1.0 at given time intervals otherwise returns 0.0.
    See https://www.vensim.com/documentation/fn_pulse_train.html for additional info."""
    ## within timeframe
    if((current*lengthTimeStep >=start) & 
       (current*lengthTimeStep<=end) &
       ((current*lengthTimeStep)%steps == 0)):
        return 1
    else:
        return 0

def Ramp(current,growth,start,end):
    """Returns 0 until the start time and then slopes upward until end time and then holds constant."""
    if((current*lengthTimeStep>start) & (current*lengthTimeStep<end)):
        return growth*((current*lengthTimeStep)-start)
    elif (current*lengthTimeStep<=start):
        return 0
    else:
        return growth*((end*lengthTimeStep)-start)

delaylist = np.zeros(len(timeSteps))
def DelayFixed(current,inp,delayTime,initialValue): 
    """Returns 0 until the startTime and then slopes upward until endTime and then holds constant."""
    delaylist[current] = inp
    if current*lengthTimeStep<=delayTime:
        return initialValue
    else:
        return delaylist[int(current-(delayTime/lengthTimeStep))]
    
def LookUp(val,lookup):
    """Write a description of this function here for personal documentation."""
    lookupin = lookup[0]
    lookupout = lookup[1]
    return np.interp(val,lookupin,lookupout)


## Help functions to chech your results only needed from model 3.2 onwards
def CheckResults(version,actProductPrice, capitalStock,productOutput):
    df = pd.read_excel('SD Assignment Check Results 2019.xlsx', sheet_name='Check values')
    columnName = [col for col in df if col.startswith(version)][0]
    columnNumber = df.columns.get_loc(columnName)
    rows = df.iloc[1:101,columnNumber:columnNumber+3].values
    difference = False
    for i in range(0,100):
        checkProductPrice = CheckDifferentValue(rows[i][0],actProductPrice[i])
        checkCaptialStock = CheckDifferentValue(rows[i][1],capitalStock[i])
        checkProductOutput = CheckDifferentValue(rows[i][2],productOutput[i])
        if(checkProductPrice):
            print("Difference of",rows[i][0] - actProductPrice[i], "at time",i+1,"for" ,columnName,", ActProductPrice")
        if(checkCaptialStock):
            print("Difference of",rows[i][1] - capitalStock[i], "at time",i+1,"for",columnName,", CapitalStock")
        if(checkProductOutput):
            print("Difference of",rows[i][2] - productOutput[i], "at time",i+1,"for",columnName,", ProductOutput")
        difference = difference | checkProductPrice | checkCaptialStock | checkProductOutput
    if(difference):
        print("Check Completed, some values are differnt. See messages above")
    else :
        print("Check Completed, all values are the same")

     
def CheckDifferentValue(value1, value2):
    differenceThreshold = 0.001
    return abs(value1 - value2) >differenceThreshold

#%% Input variables


"""Stocks"""
# 1.0 Initial 
CapitalStock = np.zeros(len(timeSteps))*np.nan
CapitalStock[0] = 100 # for stock you will need to define the initial value
# 2.0 
# 3.1
# 3.2
# 3.3 
CumulativeProduction = np.zeros(len(timeSteps))*np.nan
CumulativeProduction[0] = 1


"""Variables"""
# 1.0 Initial
InvestmentExogenous                 = np.zeros(len(timeSteps))*np.nan
# 2.0 
DesiredProductOutput                = np.zeros(len(timeSteps))*np.nan
ProductOutput                       = np.zeros(len(timeSteps))*np.nan
RequiredCSNetInvestment             = np.zeros(len(timeSteps))*np.nan
# 3.1
PriceEnergy                         = np.zeros(len(timeSteps))*np.nan
RefProductCost                      = np.zeros(len(timeSteps))*np.nan
# 3.2
ScalingMultiplier                   = np.zeros(len(timeSteps))*np.nan
OverallCostMultiplier               = np.zeros(len(timeSteps))*np.nan
ActProductPrice                     = np.zeros(len(timeSteps))*np.nan
ProdDemMultiplier                   = np.zeros(len(timeSteps))*np.nan
# 3.3 
LearningMultiplier                  = np.zeros(len(timeSteps))*np.nan
ProductionIn                        = np.zeros(len(timeSteps))*np.nan


"""Flow"""
# 1.0 Initial 
InvestmentFlow                      = np.zeros(len(timeSteps))*np.nan
DepreciationFlow                    = np.zeros(len(timeSteps))*np.nan
# 2.0 
InvestmentFlow                      = np.zeros(len(timeSteps))*np.nan
# 3.1
# 3.2
# 3.3 etc.



"""Constants"""
# 1.0 Initial
DepreciationCS = 0.1
PulseSize = 5
PulseInterval = 5
# 2.0 
LinearDemandGrowthforP = 0
OKR = 1
DelayPeriod = 0.25
# 3.1
OER = 1
LinearENPriceIncrease = 0
PriceCapital = 0.1

# 3.2
ScalingFactor = 0.66
RefSizeCS = 100
InitialPriceP = 0.22
ProdPriceElasticity = -0.78
# 3.3 
LearnCoefficient = 0



#%% Main Loop

for t in range(len(timeSteps)):
    #Calculate stocks, make sure to multiply the flows with the lengthTimeStep!
    if t != 0: # Initial value already defined!
        
        """Stocks"""
        # 1.0 Initial
        CapitalStock[t] = CapitalStock[t-1]+(InvestmentFlow[t-1]-DepreciationFlow[t-1])*lengthTimeStep
        # 2.0
        # 3.0 etc
        # Add other Stocks here
        # Stock[t] = Stock[t-1]
        CumulativeProduction[t] = CumulativeProduction [t-1]+ (ProductionIn[t-1])*lengthTimeStep
        
    
    """Variables"""
    # 1.0 Initial
    InvestmentExogenous[t] = PulseSize*PulseTrain(t,10,PulseInterval,300)
    # 2.0 
  
    ProductOutput[t] = CapitalStock[t]*OKR
    # 3.0 etc.
    PriceEnergy[t] = 0.1 + Ramp(t, LinearENPriceIncrease, 10, 300)
    RefProductCost[t]= (PriceCapital/OKR) + (PriceEnergy[t]/OER)
    # 3.2
    ScalingMultiplier[t] = (CapitalStock[t]/RefSizeCS)**(-ScalingFactor)
    OverallCostMultiplier[t] = ScalingMultiplier[t]
    ActProductPrice[t]= 1.1 * RefProductCost[t]*OverallCostMultiplier[t]
    ProdDemMultiplier[t] = 1 + ProdPriceElasticity*((ActProductPrice[t]-InitialPriceP)/InitialPriceP)
    DesiredProductOutput[t] = (105 + Ramp(t, LinearDemandGrowthforP, 10, 300))*ProdDemMultiplier[t]
    RequiredCSNetInvestment[t] = (DesiredProductOutput[t]-ProductOutput[t])/OKR
    #3.3
    ProductionIn[t] = ProductOutput[t]
    
    """Flow"""
    # 1.0 Initial
    DepreciationFlow[t] = CapitalStock[t]*DepreciationCS
    InvestmentFlow[t] = max(0, (RequiredCSNetInvestment[t]+ DepreciationFlow[t])) # The max function is used to avoid negative values.
    # 2.0
   
    # 3.0 etc. 
    
    



#%% Generate output
data = [CapitalStock,InvestmentFlow,DepreciationFlow,DesiredProductOutput,RefProductCost]
data = np.transpose(data)
label = ['Capital Stock','InvestFlow','DepreciationFlow','DesiredProductOutput','RefProductCost']
out = pd.DataFrame(data,index=timeSteps,columns=label)


out.to_excel('mdl1.xlsx')
out.plot(lw = 2, kind='line', colormap='jet', subplots=True, grid=True, figsize=(5,8), xlim=[initialTime,finalTime], title='Results')

# Only uncomment lines below when you want to check your results 
#CheckResults("3.2", ActProductPrice,CapitalStock,ProductOutput)
#CheckResults("3.3", ActProductPrice,CapitalStock,ProductOutput)
#CheckResults("3.4", ActProductPrice,CapitalStock,ProductOutput)
#CheckResults("4", ActProductPrice,CapitalStock,ProductOutput)
#CheckResults("5", ActProductPrice,CapitalStock,ProductOutput)
#CheckResults("6", ActProductPrice,CapitalStock,ProductOutput)
#CheckResults("7", ActProductPrice,CapitalStock,ProductOutput)
