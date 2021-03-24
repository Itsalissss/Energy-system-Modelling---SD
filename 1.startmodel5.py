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
    
def LookUp(val,lookupin,lookupout):
    """Write a description of this function here for personal documentation."""
    lookupin = fossilin 
    lookupout = fossilout
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
CumulativeProduction                = np.zeros(len(timeSteps))*np.nan
CumulativeProduction[0] = 1
#4.0 Fossil fuels
UltOilReserves                      = np.zeros(len(timeSteps))*np.nan
UltOilReserves[0]                   = 10000
OilProductionReserves               = np.zeros(len(timeSteps))*np.nan
OilProductionReserves[0]            = 2000
CumulativeUse                       = np.zeros(len(timeSteps))*np.nan
CumulativeUse[0]                    = 20
# 5 Biofuel
BiomassProductionLand               = np.zeros(len(timeSteps))*np.nan
BiomassProductionLand[0]            = 1 #ha
CumulatedBioConv                    = np.zeros(len(timeSteps))*np.nan
CumulatedBioConv[0]                 = 10
BiofuelConvCapital                  = np.zeros(len(timeSteps))*np.nan
BiofuelConvCapital[0]               = 10


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
# 3.4 Production economics: subtitution
InitialProductCost                  = np.zeros(len(timeSteps))*np.nan
OptProductCost                      = np.zeros(len(timeSteps))*np.nan
OptEnergyUse                        = np.zeros(len(timeSteps))*np.nan
OptCapitalStock                     = np.zeros(len(timeSteps))*np.nan
OptOER                              = np.zeros(len(timeSteps))*np.nan
OptOKR                              = np.zeros(len(timeSteps))*np.nan
EnergyCostMultiplier                = np.zeros(len(timeSteps))*np.nan
CapEnSubstMultiplier                = np.zeros(len(timeSteps))*np.nan
EnergyDemand                        = np.zeros(len(timeSteps))*np.nan
# 4.0 Fossil Fuels
DepletionCostMultiplier             = np.zeros(len(timeSteps))*np.nan
PriceOfOil                          = np.zeros(len(timeSteps))*np.nan

# 5 Biofuel
DemandForBiomass                    = np.zeros(len(timeSteps))*np.nan
Yield                               = np.zeros(len(timeSteps))*np.nan
BiomassCost                         = np.zeros(len(timeSteps))*np.nan
BiomassProduction                   = np.zeros(len(timeSteps))*np.nan
CostOfBiofuel                       = np.zeros(len(timeSteps))*np.nan
BioConvLearningMultiplier           = np.zeros(len(timeSteps))*np.nan
BioConvOKR                          = np.zeros(len(timeSteps))*np.nan
BioConvCost                         = np.zeros(len(timeSteps))*np.nan
BiofuelProduction                   = np.zeros(len(timeSteps))*np.nan
RequiredInvestment                  = np.zeros(len(timeSteps))*np.nan
DemandForBiofuel                    = np.zeros(len(timeSteps))*np.nan

"""Flow"""
# 1.0 Initial 
InvestmentFlow                      = np.zeros(len(timeSteps))*np.nan
DepreciationFlow                    = np.zeros(len(timeSteps))*np.nan
# 2.0 
InvestmentFlow                      = np.zeros(len(timeSteps))*np.nan
# 3.1
# 3.2
# 3.3 etc.
# 4 Fossil Fuels
DiscoveryRate                       = np.zeros(len(timeSteps))*np.nan
ProductionRate                      = np.zeros(len(timeSteps))*np.nan
# 5 Biofuel
BioConvDepreciation                 = np.zeros(len(timeSteps))*np.nan
BioConvInvestment                   = np.zeros(len(timeSteps))*np.nan
BioConvIn                           = np.zeros(len(timeSteps))*np.nan
LandChangeRequired                  = np.zeros(len(timeSteps))*np.nan

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
PriceCapital = 0.1
# 3.2
ScalingFactor = 0
RefSizeCS = 100
InitialPriceP = 0.22
ProdPriceElasticity = -0.78
# 3.3 
LearnCoefficient = 0
#3.4
InitialOKR                          = 1
PriceOfCapital                      = 0.1
InitialOER                          = 1
InitialPrice                        = 0.22
RefSizeCS                           = 100
Alpha                               = 0.8
InitialOptEnergyUse                 = 100
InitialPriceOfEnergy                = InitialOER/InitialOKR*PriceOfCapital*(1-Alpha)/Alpha
InitialProductCost                  = PriceOfCapital/InitialOKR + InitialPriceOfEnergy/InitialOER
# 4 Fossil Fuels
DiscoveryRateFraction               = 0.01
MaxProdRateAsFrac                   = 0.1
MarketShareOil                      = 0.5
InitialUltOilReserves               = UltOilReserves[0]
# help arrays / look-up table oil reserves 
fossilin                            = np.arange(0,1.1,0.1)
fossilout                           = np.asarray([1.1,1.15,1.3,1.7,2.25,3.7,5.55,8,9.5,10,10])
# 5 Biofuel
AverageBioConvLifetime              = 20
BioConvOKRNom                       = 0.5
BiomassToBiofuelConvEff             = 0.38
LandPrice                           = 0.01 #eur/ha
LearnCoeffBio                       = 0.05
MaxLandAvailable                    = 10000
BioConvEff                          = 0.38
MarketShareBiomass                  = 0.5
InitialBiomassProductionLand        = BiomassProductionLand[0]  
InitialEnergyDemand                 = 10
#Add new help arrays
biolandin                           = np.arange(0,1.1,0.1)
biolandout                          = np.asarray([0.4,0.4,0.398,0.39,0.382,0.374,0.364,0.356,0.348,0.366,0.322])


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
        CumulativeProduction[t] = CumulativeProduction[t-1] + (ProductionIn[t-1])*lengthTimeStep
        # 4 Fossil fuel 
        UltOilReserves[t]         = UltOilReserves[t-1]+(-DiscoveryRate[t-1])*lengthTimeStep
        OilProductionReserves[t]  = OilProductionReserves[t-1]+(DiscoveryRate[t-1]-ProductionRate[t-1])*lengthTimeStep
        CumulativeUse[t]          = CumulativeUse[t-1]+(ProductionRate[t-1])*lengthTimeStep
        # 5 Biofuel
        CumulatedBioConv[t]           = CumulatedBioConv[t-1]+(BioConvIn[t-1])*lengthTimeStep
        BiofuelConvCapital[t]         = BiofuelConvCapital[t-1]+(BioConvInvestment[t-1]-BioConvDepreciation[t-1])*lengthTimeStep
        BiomassProductionLand[t]      = BiomassProductionLand[t-1]+(max(0,LandChangeRequired[t-1]))*lengthTimeStep
    
    """Variables"""
    # 1.0 Initial
    InvestmentExogenous[t] = PulseSize*PulseTrain(t,10,PulseInterval,300)
    # 2.0 
    # 3.0 etc.
    # 3.2
    ScalingMultiplier[t] = (CapitalStock[t]/RefSizeCS)**(-ScalingFactor)
    
    #3.3
    LearningMultiplier[t]= CumulativeProduction[t]**(-LearnCoefficient)
    ProductionIn[t]                = ProductOutput[t]
    #3.4 Production economics: substitution
    #4.0 Fossil Fuels
    DepletionCostMultiplier[t]     = LookUp(CumulativeUse[t]/InitialUltOilReserves, fossilin, fossilout)
    PriceOfOil[t]                  = 0.0227*DepletionCostMultiplier[t]
    PriceEnergy[t]                 = PriceOfOil[t]
    OptEnergyUse[t]                = (PriceEnergy[t] /InitialPriceOfEnergy)**(-Alpha)*InitialOptEnergyUse
    OptCapitalStock[t]             = (Alpha/(1-Alpha))*(PriceEnergy[t] /PriceOfCapital)*OptEnergyUse[t] 
    OptOER[t]                      = RefSizeCS/OptEnergyUse[t] 
    OptOKR[t]                      = RefSizeCS/OptCapitalStock[t] 
    RefProductCost[t]              = PriceOfCapital/InitialOKR + PriceEnergy[t] /InitialOER # Adapted from 3.1
    OptProductCost[t]              = PriceOfCapital/OptOKR[t]  + PriceEnergy[t] /OptOER[t] 
    EnergyCostMultiplier[t]        = RefProductCost[t]/InitialProductCost
    CapEnSubstMultiplier[t]        = OptProductCost[t] /InitialProductCost
    ProductOutput[t]               = CapitalStock[t] *OptOKR[t]  # From 2.0
    
    EnergyDemand[t]                = ProductOutput[t] /OptOER[t]
    OverallCostMultiplier[t]       = ScalingMultiplier[t]*LearningMultiplier[t]*CapEnSubstMultiplier[t]*EnergyCostMultiplier[t]
    ActProductPrice[t]             = 1.1 * InitialProductCost*OverallCostMultiplier[t]
    ProdDemMultiplier[t]           = 1 + ProdPriceElasticity*((ActProductPrice[t]-InitialPriceP)/InitialPriceP)
    DesiredProductOutput[t]        = (105 + Ramp(t, LinearDemandGrowthforP, 10, 300))*ProdDemMultiplier[t]
    RequiredCSNetInvestment[t]     = (DesiredProductOutput[t]-ProductOutput[t])/OptOKR[t]
    # 5 Biofuel
    BioConvLearningMultiplier[t]   = CumulatedBioConv[t]**(LearnCoeffBio)
    Yield[t]                       = LookUp(BiomassProductionLand[t]/MaxLandAvailable,biolandin, biolandout)
    BiomassProduction[t]           = Yield[t]*BiomassProductionLand[t]
    BioConvOKR[t]                  = BioConvOKRNom*BioConvLearningMultiplier[t]
    BioConvCost[t]                 = PriceOfCapital/BioConvOKR[t]
    BiofuelProduction[t]           = max(0,BiofuelConvCapital[t]*BioConvOKR[t])
    DemandForBiomass[t]            = BiofuelProduction[t]/BiomassToBiofuelConvEff
    BiomassCost[t]                 = LandPrice/Yield[t]
    CostOfBiofuel[t]               = BioConvCost[t]+BiomassCost[t]
    # 5 Biofuel (continued)
    DemandForBiofuel[t]            = MarketShareBiomass*EnergyDemand[t]
    RequiredInvestment[t]          = (DemandForBiofuel[t]-BiofuelProduction[t])/BioConvOKR[t]
    
    """Flow"""
    # 1.0 Initial
    DepreciationFlow[t] = CapitalStock[t]*DepreciationCS
    InvestmentFlow[t] = max(0, (RequiredCSNetInvestment[t]+ DepreciationFlow[t])) # The max function is used to avoid negative values.
    # 2.0
    # 3.0 etc. 
    # 4 Fossil fuel 
    ProductionRate[t]             = min(MaxProdRateAsFrac*OilProductionReserves[t],MarketShareOil*EnergyDemand[t])
    DiscoveryRate[t]              = DiscoveryRateFraction*UltOilReserves[t]
    # 5 Biofuel
    BioConvDepreciation[t]        = BiofuelConvCapital[t]/AverageBioConvLifetime
    BioConvInvestment[t]          = RequiredInvestment[t]+BioConvDepreciation[t]
    BioConvIn[t]                  = BiofuelProduction[t]
    LandChangeRequired[t]         = DemandForBiomass[t]/Yield[t]
    



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
