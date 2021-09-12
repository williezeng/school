# Spanning Tree project for GA Tech OMS-CS CS 6250 Computer Networks
#
# This defines a Switch that can can send and receive spanning tree 
# messages to converge on a final loop free forwarding topology.  This
# class is a child class (specialization) of the StpSwitch class.  To 
# remain within the spirit of the project, the only inherited members
# functions the student is permitted to use are:
#
# self.switchID                   (the ID number of this switch object)
# self.links                      (the list of swtich IDs connected to this switch object)
# self.send_message(Message msg)  (Sends a Message object to another switch)
#
# Student code MUST use the send_message function to implement the algorithm - 
# a non-distributed algorithm will not receive credit.
#
# Student code should NOT access the following members, otherwise they may violate
# the spirit of the project:
#
# topolink (parameter passed to initialization function)
# self.topology (link to the greater topology structure used for message passing)
#
# Copyright 2016 Michael Brown, updated by Kelly Parks
#           Based on prior work by Sean Donovan, 2015, updated for new VM by Jared Scott and James Lohse

from Message import *
from StpSwitch import *


class Switch(StpSwitch):

    def __init__(self, idNum, topolink, neighbors):
        # Invoke the super class constructor, which makes available to this object the following members:
        # -self.switchID                   (the ID number of this switch object)
        # -self.links                      (the list of swtich IDs connected to this switch object)
        super(Switch, self).__init__(idNum, topolink, neighbors)
        self.rootID = self.switchID
        self.distance = 0
        self.switchthrough = self.switchID
        self.active_links = {neighbor:True for neighbor in neighbors} # will be overwritten anyway as the messages come

    def send_initial_messages(self):
        for neighbor in self.links:
            message = Message(self.rootID, self.distance, self.switchID, neighbor, False)
            self.send_message(message)
        return

    def process_message(self, message):
        if message.root < self.rootID:
            self.rootID = message.root
            self.distance = message.distance + 1
            self.switchthrough = message.origin
            pass_throu_map = {neigh: (True if neigh == self.switchthrough else False) for neigh in self.links}
            for neighbour in self.links:
                msg = Message(self.rootID, self.distance, self.switchID, neighbour, pass_throu_map[neighbour])
                self.send_message(msg)
        # after many messages have been sent, video 35:00
        elif message.root == self.rootID:
            if message.distance + 1 < self.distance:
                self.distance = message.distance + 1
                self.switchthrough = message.origin
                pass_throu_map = {neigh: (True if neigh == self.switchthrough else False) for neigh in self.links}
                for neighbour in self.links:
                    msg = Message(self.rootID, self.distance, self.switchID, neighbour, pass_throu_map[neighbour])
                    self.send_message(msg)
            elif message.distance + 1 == self.distance:
                if message.origin < self.switchthrough:
                    self.active_links[self.switchthrough] = False
                    self.switchthrough = message.origin
                    pass_throu_map = {neigh:(True if neigh == self.switchthrough else False) for neigh in self.links}
                    for neighbour in self.links:
                        msg = Message(self.rootID, self.distance, self.switchID, neighbour, pass_throu_map[neighbour])
                        self.send_message(msg)
                elif message.origin > self.switchthrough:
                    self.active_links[message.origin] = False
                    pass_throu_map = {neigh:(True if neigh == self.switchthrough else False) for neigh in self.links}
                    for neighbour in self.links:
                        msg = Message(self.rootID, self.distance, self.switchID, neighbour, pass_throu_map[neighbour])
                        self.send_message(msg)
            else:
                if message.pathThrough is True:
                    self.active_links[message.origin] = True
                else:
                    self.active_links[message.origin] = False


    def generate_logstring(self):
        sorted_log_string = ["{first} - {second}".format(first=self.switchID, second=key) for key, value in sorted(self.active_links.items()) if value]
        return ', '.join(sorted_log_string)
