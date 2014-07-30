//
//  CSVParser.h
//  SimpelGlKitGame
//
//  Created by Mohammadreza Hosseini on 24/06/13.
//  Copyright (c) 2013 Mohammadreza Hosseini. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface CSVParser : NSObject


+ (NSArray *)parseCSVIntoArrayOfArraysFromFile:(NSString *)path
                  withSeparatedCharacterString:(NSString *)character
                          quoteCharacterString:(NSString *)quote;

@end
