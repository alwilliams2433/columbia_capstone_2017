scoring_df <- read.csv("~/Desktop/Git/columbia_capstone_2017/nyc-check-ins/scoring_df.csv")

g<- ggplot(scoring_df, aes(x=subway))
g + geom_histogram(binwidth = binsize, fill="violet", color = "black", origin = -10) + ggtitle("Figure: Histogram of Neighborhoods by Number of Subways") + xlab("Bins by Number of Subways") + ylab("Count of Neighborhoods")

